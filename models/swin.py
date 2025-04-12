import torch
from torch import nn
from torchvision.models import swin_t, Swin_T_Weights, swin_s, Swin_S_Weights,  swin_b, Swin_B_Weights
from torchvision.ops.misc import Permute

class SwinEncoder(nn.Module):
    def __init__(self, pretrained = True, freeze_backbone = True):
        super().__init__()

        # Load pre-trained Swin model
        if pretrained:
            weights = Swin_T_Weights.IMAGENET1K_V1
        else:
            weights = None
        swin = swin_t(weights = weights, progress = True)
        # print(swin)

        self.encoder = swin.features

        self.extract1 = self.create_extractor(96, 256)
        self.extract2 = self.create_extractor(192, 512)
        self.extract3 = self.create_extractor(384, 1024)
        self.extract4 = self.create_extractor(768, 2048)

        # Freeze the backbone
        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Initialize extractors
        for extractor in [self.extract1, self.extract2, self.extract3, self.extract4]:
            for layer in extractor:
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

    def create_extractor(self, in_channels, out_channels):
        return nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.ReLU(inplace=True),
            Permute([0, 3, 1, 2]),
            # nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            # Permute([0, 2, 3, 1]), 
            # torch.nn.LayerNorm(out_channels),
            # Permute([0, 3, 1, 2]),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        features = []

        extractors = [self.extract1, self.extract2, self.extract3, self.extract4]
        extraction_indices = {1, 3, 5, 7}
        extractor_idx = 0
        
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i in extraction_indices:
                features.append(extractors[extractor_idx](x))
                extractor_idx += 1
        
        return features
    
# Example usage
if __name__ == "__main__":
    model = SwinEncoder(pretrained = True, freeze_backbone = True)
    dummy_input = torch.randn(1, 3, 608, 608)
    fmaps = model(dummy_input)
    for fmap in fmaps:
        print(fmap.shape)