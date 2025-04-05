import torch.nn as nn
import numpy as np
import torch
from .model_parts import CombinationModule
from . import resnet
import mmcv
from .mmsegmentation.mmseg.apis import init_model

config_file_V1 = './models/mmsegmentation/V1.py'
checkpoint_file_V1 = './models/mmsegmentation/deeplabv3_r18b-d8_769x769_80k_cityscapes_20201225_094144-fdc985d9.pth'
config_file_V2 = './models/mmsegmentation/V2.py'
config_file_V3 = './models/mmsegmentation/V3.py'
checkpoint_file_V3 = './models/mmsegmentation/deeplabv3plus_r101-d8_769x769_80k_cityscapes_20220406_154720-dfcc0b68.pth'
config_file_V4 = './models/mmsegmentation/V4.py'
checkpoint_file_V4 = './models/mmsegmentation/pspnet_r18-d8_769x769_80k_cityscapes_20201225_021458-3deefc62.pth'
config_file_V5 = './models/mmsegmentation/V5.py'
checkpoint_file_V5 = './models/mmsegmentation/deeplabv3plus_r18-d8_769x769_80k_cityscapes_20201226_083346-f326e06a.pth'
#V5: mim download mmsegmentation --config deeplabv3plus_r18-d8_4xb2-80k_cityscapes-769x769 --dest .
config_file_V6 = './models/mmsegmentation/V6.py'

#V1 is cascading deeplabv3 and resnet152
#V2 is using deeplabv3 as base network for feature map
#V3 is using deeplabv3+ as base network for feature map
#V4 is using pspnet as base network for feature map


# seg_model = init_model(config_file, checkpoint_file, device='mps')
if (torch.cuda.is_available()):
    device = torch.device('cuda')
elif (torch.backends.mps.is_available()):
    device = torch.device('mps')
else:
    device = torch.device('cpu')
class CTRBOX_mmsegmentationV1(nn.Module):
    def __init__(self, heads, pretrained, down_ratio, final_kernel, head_conv):
        super().__init__()
        channels = [3, 64, 256, 512, 1024, 2048]
        assert down_ratio in [2, 4, 8, 16]
        self.l1 = int(np.log2(down_ratio))
        self.seg_model = init_model(config_file_V1, checkpoint_file_V1, device=device)
        self.seg_model.eval()
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(3, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(3, 3, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(3, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
        )
        self.base_network = resnet.resnet152(pretrained=pretrained)

        self.dec_c2 = CombinationModule(512, 256, batch_norm=True)
        self.dec_c3 = CombinationModule(1024, 512, batch_norm=True)
        self.dec_c4 = CombinationModule(2048, 1024, batch_norm=True)
        self.heads = heads

        for head in self.heads:
            classes = self.heads[head]
            if head == 'wh':
                fc = nn.Sequential(nn.Conv2d(channels[self.l1], head_conv, kernel_size=7, padding=3, bias=True),
                                   nn.BatchNorm2d(head_conv),   # BN not used in the paper, but would help stable training
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, classes, kernel_size=7, padding=3, bias=True))
            else:
                fc = nn.Sequential(nn.Conv2d(channels[self.l1], head_conv, kernel_size=3, padding=1, bias=True),
                                   nn.BatchNorm2d(head_conv),   # BN not used in the paper, but would help stable training
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, classes, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True))
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                self.fill_fc_weights(fc)

            self.__setattr__(head, fc)
        # x = self.base_network()
        # print_layers.print_layers(self)

    def fill_fc_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # print('x:', x.shape)
        x = self.seg_model.forward(x, mode='tensor')
        # print('x segmented:', x.shape)
        x = self.upsample(x)
        # print('x upscaled:', x.shape)
        x = self.base_network(x)
        # x = x.contiguous()
        # print('x base network:', x[-1].shape)

        c4_combine = self.dec_c4(x[-1], x[-2])
        # c4_combine = c4_combine.contiguous()
        c3_combine = self.dec_c3(c4_combine, x[-3])
        # c3_combine = c3_combine.contiguous()
        c2_combine = self.dec_c2(c3_combine, x[-4])
        # c2_combine = c2_combine.contiguous()
        # print('c2_combine:', c2_combine.shape)
        
        dec_dict = {}
        for head in self.heads:
            dec_dict[head] = self.__getattr__(head)(c2_combine)
            if 'hm' in head or 'cls' in head:
                dec_dict[head] = torch.sigmoid(dec_dict[head])
        return dec_dict
class CTRBOX_mmsegmentationV2(nn.Module):
    def __init__(self, heads, pretrained, down_ratio, final_kernel, head_conv):
        super().__init__()
        channels = [3, 64, 256, 512, 1024, 2048]
        assert down_ratio in [2, 4, 8, 16]
        self.l1 = int(np.log2(down_ratio))
        self.seg_model = init_model(config_file_V2, checkpoint_file_V1, device=device)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
        )
        self.heads = heads

        for head in self.heads:
            classes = self.heads[head]
            if head == 'wh':
                fc = nn.Sequential(nn.Conv2d(channels[self.l1], head_conv, kernel_size=7, padding=3, bias=True),
                                #    nn.BatchNorm2d(head_conv),   # BN not used in the paper, but would help stable training
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, classes, kernel_size=7, padding=3, bias=True))
            else:
                fc = nn.Sequential(nn.Conv2d(channels[self.l1], head_conv, kernel_size=3, padding=1, bias=True),
                                #    nn.BatchNorm2d(head_conv),   # BN not used in the paper, but would help stable training
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, classes, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True))
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                self.fill_fc_weights(fc)

            self.__setattr__(head, fc)
        # x = self.base_network()
        # print_layers.print_layers(self)

    def fill_fc_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # print('input x:', x.shape)
        x = self.seg_model.forward(x, mode='tensor')
        # print('x segmented:', x.shape)
        x = self.upsample(x)
        # print('x upscaled:', x.shape)
        dec_dict = {}
        for head in self.heads:
            dec_dict[head] = self.__getattr__(head)(x)
            if 'hm' in head or 'cls' in head:
                dec_dict[head] = torch.sigmoid(dec_dict[head])
        return dec_dict
class CTRBOX_mmsegmentationV3(nn.Module):
    def __init__(self, heads, pretrained, down_ratio, final_kernel, head_conv):
        super().__init__()
        channels = [3, 64, 256, 512, 1024, 2048]
        assert down_ratio in [2, 4, 8, 16]
        self.l1 = int(np.log2(down_ratio))
        self.seg_model = init_model(config_file_V3, checkpoint_file_V3, device=device)
        self.heads = heads

        for head in self.heads:
            classes = self.heads[head]
            if head == 'wh':
                fc = nn.Sequential(nn.Conv2d(channels[self.l1], head_conv, kernel_size=7, padding=3, bias=True),
                                #    nn.BatchNorm2d(head_conv),   # BN not used in the paper, but would help stable training
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, classes, kernel_size=7, padding=3, bias=True))
            else:
                fc = nn.Sequential(nn.Conv2d(channels[self.l1], head_conv, kernel_size=3, padding=1, bias=True),
                                #    nn.BatchNorm2d(head_conv),   # BN not used in the paper, but would help stable training
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, classes, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True))
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                self.fill_fc_weights(fc)

            self.__setattr__(head, fc)
        # x = self.base_network()
        # print_layers.print_layers(self)

    def fill_fc_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # print('input x:', x.shape)
        x = self.seg_model.forward(x, mode='tensor')
        # print('x segmented:', x.shape)
        
        dec_dict = {}
        for head in self.heads:
            dec_dict[head] = self.__getattr__(head)(x)
            if 'hm' in head or 'cls' in head:
                dec_dict[head] = torch.sigmoid(dec_dict[head])
        return dec_dict
class CTRBOX_mmsegmentationV4(nn.Module):
    def __init__(self, heads, pretrained, down_ratio, final_kernel, head_conv):
        super().__init__()
        channels = [3, 64, 256, 512, 1024, 2048]
        assert down_ratio in [2, 4, 8, 16]
        self.l1 = int(np.log2(down_ratio))
        self.seg_model = init_model(config_file_V4, checkpoint_file_V4, device=device)
        self.heads = heads

        for head in self.heads:
            classes = self.heads[head]
            if head == 'wh':
                fc = nn.Sequential(nn.Conv2d(channels[self.l1], head_conv, kernel_size=7, padding=3, bias=True),
                                #    nn.BatchNorm2d(head_conv),   # BN not used in the paper, but would help stable training
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, classes, kernel_size=7, padding=3, bias=True))
            else:
                fc = nn.Sequential(nn.Conv2d(channels[self.l1], head_conv, kernel_size=3, padding=1, bias=True),
                                #    nn.BatchNorm2d(head_conv),   # BN not used in the paper, but would help stable training
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, classes, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True))
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                self.fill_fc_weights(fc)

            self.__setattr__(head, fc)
        # x = self.base_network()
        # print_layers.print_layers(self)

    def fill_fc_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # print('input x:', x.shape)
        x = self.seg_model.forward(x, mode='tensor')
        print('x segmented:', x.shape)
        
        dec_dict = {}
        for head in self.heads:
            dec_dict[head] = self.__getattr__(head)(x)
            if 'hm' in head or 'cls' in head:
                dec_dict[head] = torch.sigmoid(dec_dict[head])
        return dec_dict
class CTRBOX_mmsegmentationV5(nn.Module):
    def __init__(self, heads, pretrained, down_ratio, final_kernel, head_conv):
        super().__init__()
        channels = [3, 64, 256, 512, 1024, 2048]
        assert down_ratio in [2, 4, 8, 16]
        self.l1 = int(np.log2(down_ratio))
        self.seg_model = init_model(config_file_V5, checkpoint_file_V5, device=device)
        self.heads = heads

        for head in self.heads:
            classes = self.heads[head]
            if head == 'wh':
                fc = nn.Sequential(nn.Conv2d(channels[self.l1], head_conv, kernel_size=7, padding=3, bias=True),
                                #    nn.BatchNorm2d(head_conv),   # BN not used in the paper, but would help stable training
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, classes, kernel_size=7, padding=3, bias=True))
            else:
                fc = nn.Sequential(nn.Conv2d(channels[self.l1], head_conv, kernel_size=3, padding=1, bias=True),
                                #    nn.BatchNorm2d(head_conv),   # BN not used in the paper, but would help stable training
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, classes, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True))
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                self.fill_fc_weights(fc)

            self.__setattr__(head, fc)
        # x = self.base_network()
        # print_layers.print_layers(self)

    def fill_fc_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # print('input x:', x.shape)
        x = self.seg_model.forward(x, mode='tensor')
        # print('x segmented:', x.shape)
        
        dec_dict = {}
        for head in self.heads:
            dec_dict[head] = self.__getattr__(head)(x)
            if 'hm' in head or 'cls' in head:
                dec_dict[head] = torch.sigmoid(dec_dict[head])
        return dec_dict
class CTRBOX_mmsegmentationV6(nn.Module):
    def __init__(self, heads, pretrained, down_ratio, final_kernel, head_conv):
        super().__init__()
        channels = [3, 64, 256, 512, 1024, 2048]
        assert down_ratio in [2, 4, 8, 16]
        self.l1 = int(np.log2(down_ratio))
        self.seg_model = init_model(config_file_V5, checkpoint_file_V5, device=device)
        self.heads = heads
        # downchannel from 256 to 3
        self.downchannel = nn.Sequential(nn.Conv2d(256, 3, kernel_size=1, stride=1, padding=0),
                                    nn.BatchNorm2d(3),
                                    nn.ReLU(),
                                    nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(3),
                                    nn.ReLU())
        # upsample to 608x608
        self.upsample = nn.Sequential(nn.ConvTranspose2d(3, 3, kernel_size=2, stride=2, padding=0, output_padding=0),
                                    nn.BatchNorm2d(3),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(3, 3, kernel_size=2, stride=2, padding=0, output_padding=0),
                                    nn.BatchNorm2d(3),
                                    nn.ReLU())
        # mix mmsegmentation with original data
        self.images_mix = nn.Sequential(nn.Conv2d(6, 3, kernel_size=1, stride=1, padding=0),
                                    nn.BatchNorm2d(3),
                                    nn.ReLU(),
                                    nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(3),
                                    nn.ReLU())
        self.base_network = resnet.resnet152(pretrained=pretrained)
        #neck
        self.dec_c2 = CombinationModule(512, 256, batch_norm=True)
        self.dec_c3 = CombinationModule(1024, 512, batch_norm=True)
        self.dec_c4 = CombinationModule(2048, 1024, batch_norm=True)
        # auxiliary head
        self.aux_head = heads
        for head in self.aux_head:
            classes = self.aux_head[head]
            if head == 'wh':
                fc = nn.Sequential(nn.Conv2d(channels[self.l1], head_conv, kernel_size=7, padding=3, bias=True),
                                   nn.BatchNorm2d(head_conv),   # BN not used in the paper, but would help stable training
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, classes, kernel_size=7, padding=3, bias=True))
            else:
                fc = nn.Sequential(nn.Conv2d(channels[self.l1], head_conv, kernel_size=3, padding=1, bias=True),
                                   nn.BatchNorm2d(head_conv),   # BN not used in the paper, but would help stable training
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, classes, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True))
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                self.fill_fc_weights(fc)

            self.__setattr__(head, fc)
        # decoder
        for head in self.heads:
            classes = self.heads[head]
            if head == 'wh':
                fc = nn.Sequential(nn.Conv2d(channels[self.l1], head_conv, kernel_size=7, padding=3, bias=True),
                                   nn.BatchNorm2d(head_conv),   # BN not used in the paper, but would help stable training
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, classes, kernel_size=7, padding=3, bias=True))
            else:
                fc = nn.Sequential(nn.Conv2d(channels[self.l1], head_conv, kernel_size=3, padding=1, bias=True),
                                   nn.BatchNorm2d(head_conv),   # BN not used in the paper, but would help stable training
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, classes, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True))
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                self.fill_fc_weights(fc)

            self.__setattr__(head, fc)
        # x = self.base_network()
        # print_layers.print_layers(self)

    def fill_fc_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        images = x
        print('input x:', x.shape)
        x = self.seg_model.forward(x, mode='tensor')
        print('x segmented:', x.shape)
        aux_dec_dict = {}
        for head in self.aux_head:
            aux_dec_dict[head] = self.__getattr__(head)(x)
            if 'hm' in head or 'cls' in head:
                aux_dec_dict[head] = torch.sigmoid(aux_dec_dict[head])
        x = self.downchannel(x)
        x = self.upsample(x)
        print('x upscaled:', x.shape)
        print('images:', images.shape)
        x = torch.cat((x, images), dim=1)
        x = self.images_mix(x)
        x = self.base_network(x)
        for idx, layer in enumerate(x):
            print('layer {} shape: {}'.format(idx, layer
                                                .shape))
        c4_combine = self.dec_c4(x[-1], x[-2])
        c3_combine = self.dec_c3(c4_combine, x[-3])
        c2_combine = self.dec_c2(c3_combine, x[-4])
        dec_dict = {}
        dec_dict = {}
        for head in self.heads:
            dec_dict[head] = self.__getattr__(head)(c2_combine)
            if 'hm' in head or 'cls' in head:
                dec_dict[head] = torch.sigmoid(dec_dict[head])
        return dec_dict, aux_dec_dict