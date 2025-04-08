import torch.nn.functional as F
import torch.nn as nn
import torch

class CombinationModule(nn.Module):
    def __init__(self, c_low, c_up, batch_norm=False, group_norm=False, instance_norm=False):
        super(CombinationModule, self).__init__()
        if batch_norm:
            self.up =  nn.Sequential(nn.Conv2d(c_low, c_up, kernel_size=3, padding=1, stride=1),
                                     nn.BatchNorm2d(c_up),
                                     nn.ReLU(inplace=True))
            self.cat_conv =  nn.Sequential(nn.Conv2d(c_up*2, c_up, kernel_size=1, stride=1),
                                           nn.BatchNorm2d(c_up),
                                           nn.ReLU(inplace=True))
        elif group_norm:
            self.up = nn.Sequential(nn.Conv2d(c_low, c_up, kernel_size=3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=32, num_channels=c_up),
                                    nn.ReLU(inplace=True))
            self.cat_conv = nn.Sequential(nn.Conv2d(c_up * 2, c_up, kernel_size=1, stride=1),
                                          nn.GroupNorm(num_groups=32, num_channels=c_up),
                                          nn.ReLU(inplace=True))
        elif instance_norm:
            self.up = nn.Sequential(nn.Conv2d(c_low, c_up, kernel_size=3, padding=1, stride=1),
                                    nn.InstanceNorm2d(num_features=c_up),
                                    nn.ReLU(inplace=True))
            self.cat_conv = nn.Sequential(nn.Conv2d(c_up * 2, c_up, kernel_size=1, stride=1),
                                          nn.InstanceNorm2d(num_features=c_up),
                                          nn.ReLU(inplace=True))
        else:
            self.up =  nn.Sequential(nn.Conv2d(c_low, c_up, kernel_size=3, padding=1, stride=1),
                                     nn.ReLU(inplace=True))
            self.cat_conv =  nn.Sequential(nn.Conv2d(c_up*2, c_up, kernel_size=1, stride=1),
                                           nn.ReLU(inplace=True))

    def forward(self, x_low, x_up):
        x_low = self.up(F.interpolate(x_low, x_up.shape[2:], mode='bilinear', align_corners=False))
        return self.cat_conv(torch.cat((x_up, x_low), 1))
class CombinationModule_Transpose(nn.Module):
    def __init__(self, c_low, c_up):
        super().__init__()
        self.c_low = c_low
        self.c_up = c_up
        self.refine =  nn.Sequential(nn.Conv2d(c_low, c_up, kernel_size=3, padding=1, stride=1),
                                    nn.BatchNorm2d(c_up),
                                    nn.ReLU(inplace=True))
        self.up = nn.Sequential(nn.ConvTranspose2d(c_low, c_low, kernel_size=2, padding=0,stride=2),
                               nn.BatchNorm2d(c_low),
                               nn.ReLU(inplace=True))
        self.cat_conv =  nn.Sequential(nn.Conv2d(c_up*2, c_up, kernel_size=1, stride=1),
                                        nn.BatchNorm2d(c_up),
                                        nn.ReLU(inplace=True))
    def forward(self, x_low, x_up):
        x_low = self.refine(self.up(x_low))
        return self.cat_conv(torch.cat((x_up, x_low), 1))
class CombinationModule_Addition(nn.Module):
    def __init__(self, c_low, c_up):
        super().__init__()
        self.c_low = c_low
        self.c_up = c_up
        self.refine =  nn.Sequential(nn.Conv2d(c_low, c_up, kernel_size=3, padding=1, stride=1),
                                     nn.BatchNorm2d(c_up),
                                     nn.ReLU(inplace=True))
        self.mix = nn.Sequential(nn.Conv2d(c_up, c_up, kernel_size=3, padding=1, stride=1),
                                 nn.BatchNorm2d(c_up),
                                 nn.ReLU(inplace=True))
    def forward(self, x_low, x_up):
        x_low = F.interpolate(x_low, x_up.shape[2:], mode='bilinear', align_corners=False)
        x_low = self.refine(x_low)
        addition = x_low + x_up
        x_up = self.mix(addition)
        return x_up