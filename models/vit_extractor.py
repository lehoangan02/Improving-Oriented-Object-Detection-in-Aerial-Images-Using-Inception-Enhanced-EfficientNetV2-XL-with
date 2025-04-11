import torch
from torchvision.models import vit_b_16, ViT_B_16_Weights
import math
from torchvision.ops import Permute

class InputProcessor(torch.nn.Module):
    def __init__(self, vit):
        super().__init__()
        self.vit = vit
        self.cached_posemb = None  # Cache for resized positional embeddings

    def resize_pos_embed(self, posemb, posemb_new): # example: 224: (14*14 +1) --> 608: (38*38 +1)
        # Rescale the grid of position embeddings when loading from state_dict. Adapted from
        # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
        ntok_new = posemb_new.shape[1]
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]  # Separate class token and grid embeddings
        ntok_new -= 1

        # Calculate old and new grid sizes
        gs_old = int(math.sqrt(len(posemb_grid)))
        gs_new = int(math.sqrt(ntok_new))

        # Reshape and interpolate positional embeddings
        posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
        posemb_grid = torch.nn.functional.interpolate(posemb_grid, size=(gs_new, gs_new), mode='bicubic')
        posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)

        # Concatenate class token and resized grid embeddings
        posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
        return posemb

    def forward(self, x):
        n, c, h, w = x.shape

        # Flatten spatial dimensions and permute
        x = x.reshape(n, c, h * w).permute(0, 2, 1)

        # Add class token
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        # Cache resized positional embeddings if input resolution is fixed
        if self.cached_posemb is None:
            posemb = self.vit.encoder.pos_embedding
            self.cached_posemb = self.resize_pos_embed(posemb, x)  # Cache the resized embeddings

        # Add cached positional embeddings
        x = x + self.cached_posemb
        return x


class ViTExtractor(torch.nn.Module):
    def __init__(self, pretrained=True, freeze_backbone=True, unfreeze_ratio=0.0):
        super().__init__()

        # Load pre-trained ViT model
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        self.vit = vit_b_16(weights=weights, progress=True)

        # Feature extractor and encoder
        self.feature_extractor = torch.nn.Sequential(
            self.vit.conv_proj,
            InputProcessor(self.vit)
        )
        self.encoder = self.vit.encoder

        # Freeze or unfreeze layers
        if freeze_backbone:
            self.set_requires_grad(self.feature_extractor, False)
            self.set_requires_grad(self.encoder, False)
        elif unfreeze_ratio > 0.0:
            self.unfreeze_layers(unfreeze_ratio)

        # Multi-scale feature maps
        self.conv1 = self.create_conv_block(768, 256, kernel_size=8, stride=4, padding=2, transpose=True)
        self.conv2 = self.create_conv_block(768, 512, kernel_size=4, stride=2, padding=1, transpose=True)
        self.conv3 = self.create_conv_block(768, 1024, kernel_size=3, stride=1, padding=1)
        self.conv4 = self.create_conv_block(768, 2048, kernel_size=3, stride=2, padding=1)

        # Initialize layers
        self.initialize_layers([self.conv1, self.conv2, self.conv3, self.conv4])

    def set_requires_grad(self, module, requires_grad):
        for param in module.parameters():
            param.requires_grad = requires_grad

    def unfreeze_layers(self, unfreeze_ratio):
        num_layers = len(self.encoder.layers)
        unfreeze_idx = max(0, num_layers - int(unfreeze_ratio * num_layers))
        print(f"Unfreezing layers from index {unfreeze_idx} to {num_layers - 1}")
        for i, layer in enumerate(self.encoder.layers):
            self.set_requires_grad(layer, i >= unfreeze_idx)

    def create_conv_block(self, in_channels, out_channels, kernel_size, stride, padding, transpose=False):
        conv_layer = torch.nn.ConvTranspose2d if transpose else torch.nn.Conv2d
        return torch.nn.Sequential(
            conv_layer(in_channels, out_channels, kernel_size, stride, padding),
            # Permute([0, 2, 3, 1]),  # Change the order of dimensions
            # torch.nn.LayerNorm(out_channels),
            # Permute([0, 3, 1, 2]),  # Change back to original order
            torch.nn.GroupNorm(num_groups=32, num_channels=out_channels),
            torch.nn.ReLU(inplace=True),
        )

    def initialize_layers(self, layers):
        for conv in layers:
            for layer in conv:
                if isinstance(layer, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
                    torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    if layer.bias is not None:
                        torch.nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.vit.encoder.dropout(x)

        features = self.encoder.layers(x)
        features = self.encoder.ln(features)

        features = features[:, 1:, :]  # Remove class token
        batch_size, num_tokens, hidden_dim = features.shape
        grid_size = int(num_tokens ** 0.5)
        feature_map = features.reshape(batch_size, grid_size, grid_size, hidden_dim).permute(0, 3, 1, 2)

        return [self.conv1(feature_map), self.conv2(feature_map), self.conv3(feature_map), self.conv4(feature_map)]

# Example usage:
if __name__ == "__main__":
    model = ViTExtractor(pretrained=True, freeze_backbone=True)
    dummy_input = torch.randn(1, 3, 608, 608)
    fmap = model(dummy_input)
    for i, feature in enumerate(fmap):
        print(f"Feature map {i}: {feature.shape}")