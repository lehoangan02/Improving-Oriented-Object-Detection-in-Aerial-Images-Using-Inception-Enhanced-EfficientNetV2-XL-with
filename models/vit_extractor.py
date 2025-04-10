import torch
from torchvision.models import vit_b_16, ViT_B_16_Weights
import math

class InputProcessor(torch.nn.Module):
    def __init__(self, vit):
        super().__init__()
        self.vit = vit

    def resize_pos_embed(self, posemb, posemb_new): # example: 224: (14*14 +1) --> 608: (38*38 +1)
        # Rescale the grid of position embeddings when loading from state_dict. Adapted from
        # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
        ntok_new = posemb_new.shape[1]
        if True:
            posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]  # posemb_tok is for cls token, posemb_grid for the following tokens
            ntok_new -= 1
        else:
            posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
        gs_old = int(math.sqrt(len(posemb_grid)))     # 14
        gs_new = int(math.sqrt(ntok_new))             # 38
        # print('Position embedding grid-size from %s to %s', gs_old, gs_new)

        # (1, 196, hidden_dim) --> (1, 14, 14, hidden_dim) --> (1, hidden_dim, 14, 14)
        posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)  

        # (1, hidden_dim, 14, 14) --> (1, hidden_dim, 38, 38)
        posemb_grid = torch.nn.functional.interpolate(posemb_grid, size=(gs_new, gs_new), mode='bicubic') 

        # (1, hidden_dim, 38, 38) --> (1, 38*38, hidden_dim)
        posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)

        # Concat cls token and grid position embeddings --> (1, 1 + 38*38, hidden_dim)
        posemb = torch.cat([posemb_tok, posemb_grid], dim=1)   

        return posemb

    def forward(self, x):
        # x is output of conv_proj with shape: (batch_size, hidden_dim, H', W')
        n, c, h, w = x.shape  # H' = W' = image_size // patch_size

        # (n, c, h, w) --> (n, c, h*w) --> (n, h*w, c)
        x = x.reshape(n, c, h * w).permute(0, 2, 1)

        # Concat class token --> (n, 1 + h*w, c)
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        # Positional embeddings resize for different image sizes
        posemb = self.vit.encoder.pos_embedding
        posemb = self.resize_pos_embed(posemb, x)
        x = x + posemb

        return x


class ViTExtractor(torch.nn.Module):
    def __init__(self, pretrained=True, freeze_backbone=True, unfreeze_ratio=0.0):
        super().__init__()

        # Load pre-trained ViT model
        if pretrained:
            weights = ViT_B_16_Weights.IMAGENET1K_V1
        else:
            weights = None
        self.vit = vit_b_16(weights=weights, progress=True)
        
        # Separate for freezing option
        self.feature_extractor = torch.nn.Sequential(
            self.vit.conv_proj,
            InputProcessor(self.vit)
        )
        
        # Separate for freezing option
        self.encoder = self.vit.encoder
        
        # Freeze if needed
        if freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            for param in self.encoder.parameters():
                param.requires_grad = False
        elif unfreeze_ratio > 0.0:
            # Unfreeze the last unfreeze_ratio encoder layers
            num_layers = len(self.encoder.layers)
            unfreeze_idx = max(0, num_layers - int(unfreeze_ratio * num_layers))
            for i, layer in enumerate(self.encoder.layers):
                for param in layer.parameters():
                    param.requires_grad = i >= unfreeze_idx
            print(f"Unfreezing the last {num_layers - unfreeze_idx} encoder layers")

            # Unfreeze the feature extractor if needed
            for param in self.feature_extractor.parameters():
                param.requires_grad = unfreeze_ratio >= 1.0

        # Convolutions for multi-scale feature maps
        # 1/4 scale
        self.conv1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(768, 256, kernel_size=8, stride=4, padding=2),
            torch.nn.BatchNorm2d(256),
            # torch.nn.LayerNorm([152, 152]),
            torch.nn.ReLU()
        )
        # 1/8 scale
        self.conv2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(768, 512, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(512),
            # torch.nn.LayerNorm([76, 76]),
            torch.nn.ReLU()
        )
        # 1/16 scale
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(768, 1024, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(1024),
            # torch.nn.LayerNorm([38, 38]),
            torch.nn.ReLU()
        )
        # 1/32 scale
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(768, 2048, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(2048),
            # torch.nn.LayerNorm([19, 19]),
            torch.nn.ReLU()
        )

        # # Initialize multi-scale feature map convolutions
        # for conv in [self.conv1, self.conv2, self.conv3, self.conv4]:
        #     for layer in conv:
        #         if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.ConvTranspose2d):
        #             torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
        #             if layer.bias is not None:
        #                 torch.nn.init.constant_(layer.bias, 0)
    
    def forward(self, x):
        # (batch_size, channels, image_size, image_size) --> (batch_size, 1 + num_patches, hidden_dim)
        x = self.feature_extractor(x)
        x = self.vit.encoder.dropout(x)
        
        features = self.encoder.layers(x)
        features = self.encoder.ln(features)
        
        # Remove class token for detection
        features = features[:, 1:, :]  # (batch_size, num_patches, hidden_dim)
        batch_size, num_tokens, hidden_dim = features.shape
        grid_size = int(num_tokens ** 0.5)  # 14 for 196 patches
        feature_map = features.reshape(batch_size, grid_size, grid_size, hidden_dim)
        feature_map = feature_map.permute(0, 3, 1, 2)  # (batch_size, hidden_dim, grid_size, grid_size)

        # Create feature maps
        feat = []
        feat.append(self.conv1(feature_map))
        feat.append(self.conv2(feature_map))
        feat.append(self.conv3(feature_map))
        feat.append(self.conv4(feature_map))
        
        return feat

# Example usage:
if __name__ == "__main__":
    model = ViTExtractor(pretrained=True, freeze_backbone=True)
    dummy_input = torch.randn(1, 3, 608, 608)
    fmap = model(dummy_input)
    print("Feature map shape:", fmap.shape)  # Expected: (1, 768, 14, 14)
