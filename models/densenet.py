# https://github.com/andreasveit/densenet-pytorch/blob/master/densenet.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        return torch.cat([x, out], 1)

class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)

class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.avg_pool2d(out, 2)

class DenseBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, growth_rate, block, dropRate=0.0):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, dropRate)
    def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes+i*growth_rate, growth_rate, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class DenseNet3(nn.Module):
    def __init__(self, depth, num_classes, growth_rate=12,
                 reduction=0.5, bottleneck=True, dropRate=0.0):
        super(DenseNet3, self).__init__()
        in_planes = 2 * growth_rate
        n = (depth - 4) // 3
        if bottleneck:
            n //= 2

        # Initial convolution
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)

        # Dense Blocks
        self.block1 = DenseBlock(n, in_planes, growth_rate, BottleneckBlock if bottleneck else BasicBlock, dropRate)
        in_planes = in_planes + n * growth_rate
        self.trans1 = TransitionBlock(in_planes, int(math.floor(in_planes * reduction)), dropRate)
        in_planes = int(math.floor(in_planes * reduction))

        self.block2 = DenseBlock(n, in_planes, growth_rate, BottleneckBlock if bottleneck else BasicBlock, dropRate)
        in_planes = in_planes + n * growth_rate
        self.trans2 = TransitionBlock(in_planes, int(math.floor(in_planes * reduction)), dropRate)
        in_planes = int(math.floor(in_planes * reduction))

        self.block3 = DenseBlock(n, in_planes, growth_rate, BottleneckBlock if bottleneck else BasicBlock, dropRate)
        in_planes = in_planes + n * growth_rate

        # Final batch norm
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)

        # Linear layer
        self.fc = nn.Linear(in_planes, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        feat = []
        feat.append(x)  # C0
        x = self.conv1(x)
        feat.append(x)  # C1

        x = self.block1(x)
        feat.append(x)  # C2
        x = self.trans1(x)

        x = self.block2(x)
        feat.append(x)  # C3
        x = self.trans2(x)

        x = self.block3(x)
        feat.append(x)  # C4

        x = self.bn1(x)
        x = self.relu(x)
        feat.append(x)  # C5

        # NOT USED
        # x = F.adaptive_avg_pool2d(x, (1, 1))
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return feat

def _densenet(depth, num_classes, growth_rate, reduction, bottleneck, dropRate):
    return DenseNet3(depth, num_classes, growth_rate, reduction, bottleneck, dropRate)

def densenetMini(dropRate=0.0):
    # num_classes is not used
    return _densenet(depth=45, num_classes=13, growth_rate=12,
                    reduction=1.0, bottleneck=False, dropRate=0.0)
def densenet121(pretrained=False, progess=True, **kwargs):
    if pretrained:
        num_classes = 1000
    else:
        num_classes = kwargs.get('num_classes', 1000)
    return _densenet(121, num_classes, 32, 0.5, True, 0.0)

def densenet169(num_classes=1000, dropRate=0.0):
    return _densenet(169, num_classes, 32, 0.5, True, dropRate)

def densenet201(num_classes=1000, dropRate=0.0):
    return _densenet(201, num_classes, 32, 0.5, True, dropRate)

def densenet161(num_classes=1000, dropRate=0.0):
    return _densenet(161, num_classes, 48, 0.5, True, dropRate)
