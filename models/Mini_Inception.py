import torch.nn as nn
import torch
class MiniInception(nn.Module):
    def __init__(self):
        super().__init__()
        self.up1 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0), nn.ReLU()) # output: resolution is the same, channels = 64
        self.up3 = nn.Sequential(nn.Conv2d(256, 96, kernel_size=1, stride=1, padding=0), nn.ReLU(), # output: resolution is the same, channels = 96
                                nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=1), nn.ReLU()) # output: resolution is the same, channels = 128
        self.up5 = nn.Sequential(nn.Conv2d(256, 16, kernel_size=1, stride=1, padding=0), nn.ReLU(), # output: resolution is the same, channels = 16
                                nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2), nn.ReLU()) # output: resolution is the same, channels = 32
        self.maxpool = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1), # output: resolution is the same, channels = 256
                                    nn.Conv2d(256, 32, kernel_size=1, stride=1, padding=0), nn.ReLU()) # output: resolution is the same, channels = 32
    def forward(self, x):
        return torch.cat([self.up1(x), self.up3(x), self.up5(x), self.maxpool(x)], dim=1)