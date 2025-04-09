import torch.nn as nn
import numpy as np
import torch
from .model_parts import CombinationModule
from .model_parts import CombinationModule_Transpose
from .model_parts import CombinationModule_Addition
from . import resnet
from . import densenet
from . import fpn
from . import Mini_Inception as mini_inception
from . import print_layers
from . import efficientnet_v2
from . import vit_extractor
from . import resnet_fpn
import torch.nn.functional as F


class CTRBOX_Github(nn.Module):
    def __init__(self, heads, pretrained, down_ratio, final_kernel, head_conv):
        super().__init__()
        channels = [3, 64, 256, 512, 1024, 2048]
        assert down_ratio in [2, 4, 8, 16]
        self.l1 = int(np.log2(down_ratio))
        self.base_network = resnet.resnet101(pretrained=pretrained)

        self.dec_c2 = CombinationModule(512, 256, batch_norm=True)
        self.dec_c3 = CombinationModule(1024, 512, batch_norm=True)
        self.dec_c4 = CombinationModule(2048, 1024, batch_norm=True)
        self.heads = heads

        for head in self.heads:
            classes = self.heads[head]
            if head == 'wh':
                fc = nn.Sequential(nn.Conv2d(channels[self.l1], head_conv, kernel_size=3, padding=1, bias=True),
                                #    nn.BatchNorm2d(head_conv),   # BN not used in the paper, but would help stable training
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, classes, kernel_size=3, padding=1, bias=True))
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
        x = self.base_network(x)
        for idx, layer in enumerate(x):
            print('layer {} shape: {}'.format(idx, layer
                                              .shape))
        # import matplotlib.pyplot as plt
        # import os
        # for idx in range(x[1].shape[1]):
        #     temp = x[1][0,idx,:,:]
        #     temp = temp.data.cpu().numpy()
        #     plt.imsave(os.path.join('dilation', '{}.png'.format(idx)), temp)


        c4_combine = self.dec_c4(x[-1], x[-2])
        c3_combine = self.dec_c3(c4_combine, x[-3])
        c2_combine = self.dec_c2(c3_combine, x[-4])
        # print('c2_combine shape: ', c2_combine.shape)

        dec_dict = {}
        for head in self.heads:
            dec_dict[head] = self.__getattr__(head)(c2_combine)
            if 'hm' in head or 'cls' in head:
                dec_dict[head] = torch.sigmoid(dec_dict[head])
        # for dec in dec_dict:
        #     print(dec, dec_dict[dec].shape)
        return dec_dict
class CTRBOX_Paper(nn.Module):
    def __init__(self, heads, pretrained, down_ratio, final_kernel, head_conv):
        super().__init__()
        channels = [3, 64, 256, 512, 1024, 2048]
        assert down_ratio in [2, 4, 8, 16]
        self.l1 = int(np.log2(down_ratio))
        self.base_network = resnet.resnet101(pretrained=pretrained)

        self.dec_c2 = CombinationModule(512, 256, batch_norm=True)
        self.dec_c3 = CombinationModule(1024, 512, batch_norm=True)
        self.dec_c4 = CombinationModule(2048, 1024, batch_norm=True)
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

    def fill_fc_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.base_network(x)
        # for idx, layer in enumerate(x):
            # print('layer {} shape: {}'.format(idx, layer
                                            #   .shape))
        # import matplotlib.pyplot as plt
        # import os
        # for idx in range(x[1].shape[1]):
        #     temp = x[1][0,idx,:,:]
        #     temp = temp.data.cpu().numpy()
        #     plt.imsave(os.path.join('dilation', '{}.png'.format(idx)), temp)


        c4_combine = self.dec_c4(x[-1], x[-2])
        c3_combine = self.dec_c3(c4_combine, x[-3])
        c2_combine = self.dec_c2(c3_combine, x[-4])
        
        dec_dict = {}
        for head in self.heads:
            dec_dict[head] = self.__getattr__(head)(c2_combine)
            if 'hm' in head or 'cls' in head:
                dec_dict[head] = torch.sigmoid(dec_dict[head])
        return dec_dict
class CTRBOX_152(nn.Module):
    def __init__(self, heads, pretrained, down_ratio, final_kernel, head_conv):
        super().__init__()
        channels = [3, 64, 256, 512, 1024, 2048]
        assert down_ratio in [2, 4, 8, 16]
        self.l1 = int(np.log2(down_ratio))
        # self.base_network = densenet.densenet121(pretrained=pretrained)
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
        x = self.base_network(x)
        # import matplotlib.pyplot as plt
        # import os
        # for idx in range(x[1].shape[1]):
        #     temp = x[1][0,idx,:,:]
        #     temp = temp.data.cpu().numpy()
        #     plt.imsave(os.path.join('dilation', '{}.png'.format(idx)), temp)


        c4_combine = self.dec_c4(x[-1], x[-2])
        c3_combine = self.dec_c3(c4_combine, x[-3])
        c2_combine = self.dec_c2(c3_combine, x[-4])

        dec_dict = {}
        for head in self.heads:
            dec_dict[head] = self.__getattr__(head)(c2_combine)
            if 'hm' in head or 'cls' in head:
                dec_dict[head] = torch.sigmoid(dec_dict[head])
        return dec_dict
class CTRBOX_Inception(nn.Module):
    def __init__(self, heads, pretrained, down_ratio, final_kernel, head_conv):
        super().__init__()
        channels = [3, 64, 256, 512, 1024, 2048]
        assert down_ratio in [2, 4, 8, 16]
        self.l1 = int(np.log2(down_ratio))
        # self.base_network = densenet.densenet121(pretrained=pretrained)
        self.base_network = resnet.resnet152(pretrained=pretrained)
        # self.base_network = fpn.FPN101()

        self.dec_c2 = CombinationModule(512, 256, batch_norm=True)
        self.dec_c3 = CombinationModule(1024, 512, batch_norm=True)
        self.dec_c4 = CombinationModule(2048, 1024, batch_norm=True)
        self.heads = heads

        for head in self.heads:
            classes = self.heads[head]
            if head == 'wh':
                fc = nn.Sequential(mini_inception.MiniInception(),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(channels[self.l1], head_conv, kernel_size=7, padding=3, bias=True),
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


    def fill_fc_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.base_network(x)
        # import matplotlib.pyplot as plt
        # import os
        # for idx in range(x[1].shape[1]):
        #     temp = x[1][0,idx,:,:]
        #     temp = temp.data.cpu().numpy()
        #     plt.imsave(os.path.join('dilation', '{}.png'.format(idx)), temp)


        c4_combine = self.dec_c4(x[-1], x[-2])
        c3_combine = self.dec_c3(c4_combine, x[-3])
        c2_combine = self.dec_c2(c3_combine, x[-4])

        dec_dict = {}
        for head in self.heads:
            dec_dict[head] = self.__getattr__(head)(c2_combine)
            # dec_dict[head] = self.__getattr__(head)(x[self.l1])
            if 'hm' in head or 'cls' in head:
                dec_dict[head] = torch.sigmoid(dec_dict[head])
        return dec_dict
class CTRBOX_FPN(nn.Module):
    def __init__(self, heads, pretrained, down_ratio, final_kernel, head_conv):
        super().__init__()
        channels = [3, 64, 256, 512, 1024, 2048]
        assert down_ratio in [2, 4, 8, 16]
        self.l1 = int(np.log2(down_ratio))
        # self.base_network = densenet.densenet121(pretrained=pretrained)
        self.base_network = fpn.FPN101()

        self.heads = heads

        for head in self.heads:
            classes = self.heads[head]
            if head == 'wh':
                fc = nn.Sequential(mini_inception.MiniInception(),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(channels[self.l1], head_conv, kernel_size=7, padding=3, bias=True),
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
        # print_layers.print_layers(self)

    def fill_fc_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.base_network(x)
        # print('result shape: ', x[0].shape)
        # for idx, layer in enumerate(x):
            # print('layer {} shape: {}'.format(idx, layer
                                            #   .shape))
        # import matplotlib.pyplot as plt
        # import os
        # for idx in range(x[1].shape[1]):
        #     temp = x[1][0,idx,:,:]
        #     temp = temp.data.cpu().numpy()
        #     plt.imsave(os.path.join('dilation', '{}.png'.format(idx)), temp)

        dec_dict = {}
        for head in self.heads:
            dec_dict[head] = self.__getattr__(head)(x[0])
            if 'hm' in head or 'cls' in head:
                dec_dict[head] = torch.sigmoid(dec_dict[head])
        # for dec in dec_dict:
        #     print(dec, dec_dict[dec].shape)
        return dec_dict
class CTRBOX_DenseNet(nn.Module):
    def __init__(self, heads, pretrained, down_ratio, final_kernel, head_conv):
        super().__init__()
        channels = [3, 64, 256, 512, 1024, 2048]
        assert down_ratio in [2, 4, 8, 16]
        self.l1 = int(np.log2(down_ratio))
        # self.base_network = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
        self.base_network = densenet.densenetMini()
        self.adapter_layer = nn.Sequential(nn.Conv2d(492, 256, kernel_size=1, stride=1, padding=0, bias=False),
                                           nn.BatchNorm2d(256),
                                           nn.ReLU(inplace=True))
        self.dec_c2 = CombinationModule(512, 256, batch_norm=True)
        self.dec_c3 = CombinationModule(1024, 512, batch_norm=True)
        self.dec_c4 = CombinationModule(2048, 1024, batch_norm=True)
        self.heads = heads

        for head in self.heads:
            classes = self.heads[head]
            if head == 'wh':
                fc = nn.Sequential(mini_inception.MiniInception(),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(channels[self.l1], head_conv, kernel_size=7, padding=3, bias=True),
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
        # print_layers.print_layers(self)

    def fill_fc_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.base_network(x)
        # print('result shape: ', x[-1].shape)
        x = self.adapter_layer(x[-1])
        print('result shape: ', x.shape)
        dec_dict = {}
        for head in self.heads:
            dec_dict[head] = self.__getattr__(head)(x)
            if 'hm' in head or 'cls' in head:
                dec_dict[head] = torch.sigmoid(dec_dict[head])
        # for dec in dec_dict:
        #     print(dec, dec_dict[dec].shape)
        return dec_dict
class CTRBOX_EfficientNetV2(nn.Module):
    def __init__(self, heads, pretrained, down_ratio, final_kernel, head_conv):
        super().__init__()
        channels = [3, 64, 256, 512, 1024, 2048]
        assert down_ratio in [2, 4, 8, 16]
        self.l1 = int(np.log2(down_ratio))
        self.base_network = efficientnet_v2.EfficientNetV2('xl',
                        in_channels=3,
                        n_classes=256,
                        pretrained=True)
        self.adapter_layer = nn.Sequential(nn.Conv2d(32, 64, kernel_size=7, stride=2, padding=3, bias=False),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(inplace=True))
        self.dec_c2 = CombinationModule(64, 32, batch_norm=True)
        self.dec_c3 = CombinationModule(256, 64, batch_norm=True)
        self.dec_c4 = CombinationModule(640, 256, batch_norm=True)                                
        self.heads = heads

        for head in self.heads:
            classes = self.heads[head]
            if head == 'wh':
                fc = nn.Sequential(mini_inception.MiniInception(),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(channels[self.l1], head_conv, kernel_size=7, padding=3, bias=True),
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
        # print_layers.print_layers(self)

    def fill_fc_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.base_network(x)
        c4_combine = self.dec_c4(x[-1], x[-2])
        c3_combine = self.dec_c3(c4_combine, x[-3])
        c2_combine = self.dec_c2(c3_combine, x[-4])
        c2_combine = self.adapter_layer(c2_combine)
        dec_dict = {}
        for head in self.heads:
            dec_dict[head] = self.__getattr__(head)(c2_combine)
            if 'hm' in head or 'cls' in head:
                dec_dict[head] = torch.sigmoid(dec_dict[head])
        return dec_dict
class CTRBOX_ImplicitCorners(nn.Module):
    def __init__(self, heads, pretrained, down_ratio, final_kernel, head_conv):
        super().__init__()
        channels = [3, 64, 256, 512, 1024, 2048]
        assert down_ratio in [2, 4, 8, 16]
        self.l1 = int(np.log2(down_ratio))
        self.base_network = resnet.resnet152(pretrained=pretrained)

        self.dec_c2 = CombinationModule(512, 256, batch_norm=True)
        self.dec_c3 = CombinationModule(1024, 512, batch_norm=True)
        self.dec_c4 = CombinationModule(2048, 1024, batch_norm=True)
        self.heads = heads

        for head in self.heads:
            classes = self.heads[head]
            if head == 'wh':
                fc = nn.Sequential(mini_inception.MiniInception(),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(channels[self.l1], head_conv, kernel_size=7, padding=3, bias=True),
                                   nn.BatchNorm2d(head_conv),   # BN not used in the paper, but would help stable training
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, 18, kernel_size=7, padding=3, bias=True))
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

    def fill_fc_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.base_network(x)
        c4_combine = self.dec_c4(x[-1], x[-2])
        c3_combine = self.dec_c3(c4_combine, x[-3])
        c2_combine = self.dec_c2(c3_combine, x[-4])

        dec_dict = {}
        for head in self.heads:
            if head == 'wh':
                wh_and_corners = self.__getattr__(head)(c2_combine)
                wh = wh_and_corners[:, :10]
                corners = wh_and_corners[:, 10:18]
                dec_dict[head] = wh
                dec_dict['corners'] = corners
            else:
                dec_dict[head] = self.__getattr__(head)(c2_combine)
            if 'hm' in head or 'cls' in head:
                dec_dict[head] = torch.sigmoid(dec_dict[head])
        return dec_dict
class CTRBOX_ImplicitCornersV2(nn.Module):
    def __init__(self, heads, pretrained, down_ratio, final_kernel, head_conv):
        super().__init__()
        channels = [3, 64, 256, 512, 1024, 2048]
        assert down_ratio in [2, 4, 8, 16]
        self.l1 = int(np.log2(down_ratio))
        self.base_network = resnet.resnet152(pretrained=pretrained)

        self.dec_c2 = CombinationModule(512, 256, batch_norm=True)
        self.dec_c3 = CombinationModule(1024, 512, batch_norm=True)
        self.dec_c4 = CombinationModule(2048, 1024, batch_norm=True)
        self.heads = heads

        for head in self.heads:
            classes = self.heads[head]
            if head == 'wh':
                fc = nn.Sequential(mini_inception.MiniInception(),
                                   nn.ReLU(inplace=True),
                                   mini_inception.MiniInception(),
                                   nn.BatchNorm2d(head_conv),   # BN not used in the paper, but would help stable training
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, 18, kernel_size=1, padding=0, stride=1, bias=True))
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


    def fill_fc_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.base_network(x)
        c4_combine = self.dec_c4(x[-1], x[-2])
        c3_combine = self.dec_c3(c4_combine, x[-3])
        c2_combine = self.dec_c2(c3_combine, x[-4])

        dec_dict = {}
        for head in self.heads:
            if head == 'wh':
                wh_and_corners = self.__getattr__(head)(c2_combine)
                wh = wh_and_corners[:, :10]
                corners = wh_and_corners[:, 10:18]
                dec_dict[head] = wh
                dec_dict['corners'] = corners
            else:
                dec_dict[head] = self.__getattr__(head)(c2_combine)
            if 'hm' in head or 'cls' in head:
                dec_dict[head] = torch.sigmoid(dec_dict[head])
        return dec_dict
class CTRBOX_ExplicitCorners(nn.Module):
    def __init__(self, heads, pretrained, down_ratio, final_kernel, head_conv):
        super().__init__()
        channels = [3, 64, 256, 512, 1024, 2048]
        assert down_ratio in [2, 4, 8, 16]
        self.l1 = int(np.log2(down_ratio))
        self.base_network = resnet.resnet152(pretrained=pretrained)

        self.dec_c2 = CombinationModule(512, 256, batch_norm=True)
        self.dec_c3 = CombinationModule(1024, 512, batch_norm=True)
        self.dec_c4 = CombinationModule(2048, 1024, batch_norm=True)
        self.heads = heads

        for head in self.heads:
            classes = self.heads[head]
            if head == 'wh':
                fc = nn.Sequential(mini_inception.MiniInception(),
                                   nn.ReLU(inplace=True),
                                   mini_inception.MiniInception(),
                                   nn.BatchNorm2d(head_conv),   # BN not used in the paper, but would help stable training
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, 18, kernel_size=1, padding=0, stride=1, bias=True))
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


    def fill_fc_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.base_network(x)
        c4_combine = self.dec_c4(x[-1], x[-2])
        c3_combine = self.dec_c3(c4_combine, x[-3])
        c2_combine = self.dec_c2(c3_combine, x[-4])

        dec_dict = {}
        for head in self.heads:
            if head == 'wh':
                wh_and_corners = self.__getattr__(head)(c2_combine)
                wh = wh_and_corners[:, :10]
                corners = wh_and_corners[:, 10:18]
                processed_tt = wh[:, 0:2] / 2 + corners[:, 2:4] / 4 + corners[:, 4:6] / 4
                processed_rr = wh[:, 2:4] / 2 + corners[:, 4:6] / 4 + corners[:, 6:8] / 4
                processed_bb = wh[:, 4:6] / 2 + corners[:, 0:2] / 4 + corners[:, 6:8] / 4
                processed_ll = wh[:, 6:8] / 2 + corners[:, 0:2] / 4 + corners[:, 2:4] / 4
                width_height = wh[:, 8:10]
                processed_wh = torch.cat((processed_tt, processed_rr, processed_bb, processed_ll, width_height), dim=1)
                dec_dict[head] = processed_wh
            else:
                dec_dict[head] = self.__getattr__(head)(c2_combine)
            if 'hm' in head or 'cls' in head:
                dec_dict[head] = torch.sigmoid(dec_dict[head])
        return dec_dict

class CTRBOX_ViT(nn.Module):
    def __init__(self, heads, pretrained, down_ratio, final_kernel, head_conv):
        super().__init__()
        channels = [3, 64, 256, 512, 1024, 2048]
        assert down_ratio in [2, 4, 8, 16]
        self.l1 = int(np.log2(down_ratio))

        self.base_network = vit_extractor.ViTExtractor(pretrained=True, freeze_backbone=True)

        self.dec_c2 = CombinationModule(512, 256, batch_norm=True)
        self.dec_c3 = CombinationModule(1024, 512, batch_norm=True)
        self.dec_c4 = CombinationModule(2048, 1024, batch_norm=True)
        self.heads = heads

        for head in self.heads:
            classes = self.heads[head]
            if head == 'wh':
                fc = nn.Sequential(nn.Conv2d(channels[self.l1], head_conv, kernel_size=3, padding=1, bias=True),
                                   nn.BatchNorm2d(head_conv),   # BN not used in the paper, but would help stable training
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, classes, kernel_size=3, padding=1, bias=True))
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

    def fill_fc_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.base_network(x)
        # for idx, layer in enumerate(x):
            # print('layer {} shape: {}'.format(idx, layer
                                            #   .shape))
        # import matplotlib.pyplot as plt
        # import os
        # for idx in range(x[1].shape[1]):
        #     temp = x[1][0,idx,:,:]
        #     temp = temp.data.cpu().numpy()
        #     plt.imsave(os.path.join('dilation', '{}.png'.format(idx)), temp)

        # for feature in x:
        #     print('feature shape: ', feature.shape)

        c4_combine = self.dec_c4(x[-1], x[-2])
        c3_combine = self.dec_c3(c4_combine, x[-3])
        c2_combine = self.dec_c2(c3_combine, x[-4])

        dec_dict = {}
        for head in self.heads:
            dec_dict[head] = self.__getattr__(head)(c2_combine)
            if 'hm' in head or 'cls' in head:
                dec_dict[head] = torch.sigmoid(dec_dict[head])
        # for dec in dec_dict:
        #     print(dec, dec_dict[dec].shape)
        return dec_dict
class CTRBOX_Refeed_Heatmap(nn.Module):
    def __init__(self, heads, pretrained, down_ratio, final_kernel, head_conv):
        super().__init__()
        channels = [3, 64, 256, 512, 1024, 2048]
        assert down_ratio in [2, 4, 8, 16]
        self.l1 = int(np.log2(down_ratio))
        self.base_network = resnet.resnet152(pretrained=pretrained)

        self.dec_c2 = CombinationModule(512, 256, batch_norm=True)
        self.dec_c3 = CombinationModule(1024, 512, batch_norm=True)
        self.dec_c4 = CombinationModule(2048, 1024, batch_norm=True)
        self.complimentary_layer = nn.Sequential(nn.Conv2d(256, 256 - 15, kernel_size=3, padding=1, bias=True),
                                                nn.BatchNorm2d(256 - 15),
                                                nn.ReLU(inplace=True))
        self.combinational_layer = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, padding=0, bias=True),
                                                nn.BatchNorm2d(256),
                                                nn.ReLU(inplace=True))
        self.heads = heads

        for head in self.heads:
            classes = self.heads[head]
            if head == 'wh':
                fc = nn.Sequential(mini_inception.MiniInception(),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(channels[self.l1], head_conv, kernel_size=7, padding=3, bias=True),
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


    def fill_fc_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.base_network(x)

        # neck
        c4_combine = self.dec_c4(x[-1], x[-2])
        c3_combine = self.dec_c3(c4_combine, x[-3])
        c2_combine = self.dec_c2(c3_combine, x[-4])
        complementary = self.complimentary_layer(c2_combine)
        dec_dict = {}
        dec_dict['hm'] = self.__getattr__('hm')(c2_combine)
        dec_dict['hm'] = torch.sigmoid(dec_dict['hm'])
        concatenated = torch.cat((complementary, dec_dict['hm']), dim=1)
        combinational = self.combinational_layer(concatenated)

        for head in self.heads:
            if head == 'hm':
                continue
            dec_dict[head] = self.__getattr__(head)(combinational)
            if 'hm' in head or 'cls' in head:
                dec_dict[head] = torch.sigmoid(dec_dict[head])
        return dec_dict
class CTRBOX_Inception_4(nn.Module):
    def __init__(self, heads, pretrained, down_ratio, final_kernel, head_conv):
        super().__init__()
        channels = [3, 64, 256, 512, 1024, 2048]
        assert down_ratio in [2, 4, 8, 16]
        self.l1 = int(np.log2(down_ratio))
        # self.base_network = densenet.densenet121(pretrained=pretrained)
        self.base_network = resnet.resnet152(pretrained=pretrained)
        # self.base_network = fpn.FPN101()

        self.dec_c2 = CombinationModule(512, 256, batch_norm=True)
        self.dec_c3 = CombinationModule(1024, 512, batch_norm=True)
        self.dec_c4 = CombinationModule(2048, 1024, batch_norm=True)
        self.heads = heads

        for head in self.heads:
            classes = self.heads[head]
            if head == 'wh':
                fc = nn.Sequential(mini_inception.MiniInception(),
                                   nn.ReLU(inplace=True),
                                   mini_inception.MiniInception(),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, classes, kernel_size=7, padding=3, bias=True))
            else:
                fc = nn.Sequential(mini_inception.MiniInception(),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, classes, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True))
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                self.fill_fc_weights(fc)

            self.__setattr__(head, fc)


    def fill_fc_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.base_network(x)
        # import matplotlib.pyplot as plt
        # import os
        # for idx in range(x[1].shape[1]):
        #     temp = x[1][0,idx,:,:]
        #     temp = temp.data.cpu().numpy()
        #     plt.imsave(os.path.join('dilation', '{}.png'.format(idx)), temp)


        c4_combine = self.dec_c4(x[-1], x[-2])
        c3_combine = self.dec_c3(c4_combine, x[-3])
        c2_combine = self.dec_c2(c3_combine, x[-4])

        dec_dict = {}
        for head in self.heads:
            dec_dict[head] = self.__getattr__(head)(c2_combine)
            # dec_dict[head] = self.__getattr__(head)(x[self.l1])
            if 'hm' in head or 'cls' in head:
                dec_dict[head] = torch.sigmoid(dec_dict[head])
        return dec_dict
class CTRBOX_Inception_4_plus(nn.Module):
    def __init__(self, heads, pretrained, down_ratio, final_kernel, head_conv):
        super().__init__()
        channels = [3, 64, 256, 512, 1024, 2048]
        assert down_ratio in [2, 4, 8, 16]
        self.l1 = int(np.log2(down_ratio))
        # self.base_network = densenet.densenet121(pretrained=pretrained)
        self.base_network = resnet.resnet152(pretrained=pretrained)
        # self.base_network = fpn.FPN101()

        self.dec_c2 = CombinationModule(512, 256, batch_norm=True)
        self.dec_c3 = CombinationModule(1024, 512, batch_norm=True)
        self.dec_c4 = CombinationModule(2048, 1024, batch_norm=True)
        self.heads = heads

        for head in self.heads:
            classes = self.heads[head]
            if head == 'wh':
                fc = nn.Sequential(mini_inception.MiniInception(),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(channels[self.l1], head_conv, kernel_size=7, padding=3, bias=True),
                                   nn.BatchNorm2d(head_conv),   # BN not used in the paper, but would help stable training
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, classes, kernel_size=7, padding=3, bias=True))
            else:
                fc = nn.Sequential(mini_inception.MiniInception(),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(channels[self.l1], head_conv, kernel_size=3, padding=1, bias=True),
                                   nn.BatchNorm2d(head_conv),   # BN not used in the paper, but would help stable training
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, classes, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True))
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                self.fill_fc_weights(fc)

            self.__setattr__(head, fc)


    def fill_fc_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.base_network(x)
        # import matplotlib.pyplot as plt
        # import os
        # for idx in range(x[1].shape[1]):
        #     temp = x[1][0,idx,:,:]
        #     temp = temp.data.cpu().numpy()
        #     plt.imsave(os.path.join('dilation', '{}.png'.format(idx)), temp)


        c4_combine = self.dec_c4(x[-1], x[-2])
        c3_combine = self.dec_c3(c4_combine, x[-3])
        c2_combine = self.dec_c2(c3_combine, x[-4])

        dec_dict = {}
        for head in self.heads:
            dec_dict[head] = self.__getattr__(head)(c2_combine)
            # dec_dict[head] = self.__getattr__(head)(x[self.l1])
            if 'hm' in head or 'cls' in head:
                dec_dict[head] = torch.sigmoid(dec_dict[head])
        return dec_dict
class CTRBOX_Transpose_V1(nn.Module):
    def __init__(self, heads, pretrained, down_ratio, final_kernel, head_conv):
        super().__init__()
        channels = [3, 64, 256, 512, 1024, 2048]
        assert down_ratio in [2, 4, 8, 16]
        self.l1 = int(np.log2(down_ratio))
        # self.base_network = densenet.densenet121(pretrained=pretrained)
        self.base_network = resnet.resnet152(pretrained=pretrained)
        # self.base_network = fpn.FPN101()

        self.dec_c2 = CombinationModule_Transpose(512, 256)
        self.dec_c3 = CombinationModule_Transpose(1024, 512)
        self.dec_c4 = CombinationModule_Transpose(2048, 1024)
        self.heads = heads

        for head in self.heads:
            classes = self.heads[head]
            if head == 'wh':
                fc = nn.Sequential(mini_inception.MiniInception(),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(channels[self.l1], head_conv, kernel_size=7, padding=3, bias=True),
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


    def fill_fc_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.base_network(x)

        c4_combine = self.dec_c4(x[-1], x[-2])
        c3_combine = self.dec_c3(c4_combine, x[-3])
        c2_combine = self.dec_c2(c3_combine, x[-4])

        dec_dict = {}
        for head in self.heads:
            dec_dict[head] = self.__getattr__(head)(c2_combine)
            if 'hm' in head or 'cls' in head:
                dec_dict[head] = torch.sigmoid(dec_dict[head])
        return dec_dict
class CTRBOX_EfficientNetV3(nn.Module):
    def __init__(self, heads, pretrained, down_ratio, final_kernel, head_conv):
        super().__init__()
        channels = [3, 64, 256, 512, 1024, 2048]
        assert down_ratio in [2, 4, 8, 16]
        self.l1 = int(np.log2(down_ratio))
        # self.base_network = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
        self.base_network = efficientnet_v2.EfficientNetV2('xl',
                        in_channels=3,
                        n_classes=256,
                        pretrained=True)
        self.adapter_layer = nn.Sequential(nn.Conv2d(32, 64, kernel_size=7, stride=2, padding=3, bias=False),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(inplace=True))
        self.dec_c2 = CombinationModule(64, 32, group_norm=True)
        self.dec_c3 = CombinationModule(256, 64, group_norm=True)
        self.dec_c4 = CombinationModule(640, 256, group_norm=True)                                
        self.heads = heads

        for head in self.heads:
            classes = self.heads[head]
            if head == 'wh':
                fc = nn.Sequential(mini_inception.MiniInception(),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(channels[self.l1], head_conv, kernel_size=7, padding=3, bias=True),
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
        # print_layers.print_layers(self)

    def fill_fc_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.base_network(x)
        # for idx, layer in enumerate(x):
        #     print('layer {} shape: {}'.format(idx, layer.shape))
        c4_combine = self.dec_c4(x[-1], x[-2])
        c3_combine = self.dec_c3(c4_combine, x[-3])
        c2_combine = self.dec_c2(c3_combine, x[-4])
        # print('c2_combine shape: ', c2_combine.shape)
        c2_combine = self.adapter_layer(c2_combine)
        # print('c2_combine shape: ', c2_combine.shape)
        dec_dict = {}
        for head in self.heads:
            dec_dict[head] = self.__getattr__(head)(c2_combine)
            if 'hm' in head or 'cls' in head:
                dec_dict[head] = torch.sigmoid(dec_dict[head])
        return dec_dict
class CTRBOX_EfficientNetV4(nn.Module):
    def __init__(self, heads, pretrained, down_ratio, final_kernel, head_conv):
        super().__init__()
        channels = [3, 64, 256, 512, 1024, 2048]
        assert down_ratio in [2, 4, 8, 16]
        self.l1 = int(np.log2(down_ratio))
        # self.base_network = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
        self.base_network = efficientnet_v2.EfficientNetV2('xl',
                        in_channels=3,
                        n_classes=256,
                        pretrained=True)
        self.adapter_layer = nn.Sequential(nn.Conv2d(32, 64, kernel_size=7, stride=2, padding=3, bias=False),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(inplace=True))
        self.dec_c2 = CombinationModule(64, 32, group_norm=True)
        self.dec_c3 = CombinationModule(256, 64, group_norm=True)
        self.dec_c4 = CombinationModule(640, 256, group_norm=True)                                
        self.heads = heads

        for head in self.heads:
            classes = self.heads[head]
            if head == 'wh':
                fc = nn.Sequential(mini_inception.MiniInception(),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(channels[self.l1], head_conv, kernel_size=7, padding=3, bias=True),
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
        self.aux_heads = ['hm', 'wh']
        # upscale 8 times
        self.up =nn.Sequential(nn.ConvTranspose2d(640, 256, kernel_size=8, stride=8, padding=0),
                                nn.BatchNorm2d(256),
                                nn.ReLU(inplace=True))
        for head in self.aux_heads:
            if head == 'wh':
                fc = nn.Sequential(mini_inception.MiniInception(),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(channels[self.l1], head_conv, kernel_size=7, padding=3, bias=True),
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
            self.__setattr__('aux_' + head, fc)
    def fill_fc_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.base_network(x)
        aux_dict = {}
        for head in self.aux_heads:
            aux_dict[head] = self.__getattr__('aux_' + head)(self.up(x[-1]))
            if 'hm' in head or 'cls' in head:
                aux_dict[head] = torch.sigmoid(aux_dict[head])
        c4_combine = self.dec_c4(x[-1], x[-2])
        c3_combine = self.dec_c3(c4_combine, x[-3])
        c2_combine = self.dec_c2(c3_combine, x[-4])
        c2_combine = self.adapter_layer(c2_combine)
        dec_dict = {}
        for head in self.heads:
            dec_dict[head] = self.__getattr__(head)(c2_combine)
            if 'hm' in head or 'cls' in head:
                dec_dict[head] = torch.sigmoid(dec_dict[head])
        return aux_dict, dec_dict
class CTRBOX_ResnetFPN(nn.Module):
    def __init__(self, heads, pretrained, down_ratio, final_kernel, head_conv):
        super().__init__()
        print('RESNET FPN')
        channels = [3, 64, 256, 512, 1024, 2048]
        assert down_ratio in [2, 4, 8, 16]
        self.l1 = int(np.log2(down_ratio))
        # self.base_network = densenet.densenet121(pretrained=pretrained)
        self.base_network = resnet_fpn.ResnetFPN('152')

        self.dec_c2 = CombinationModule(256, 256, batch_norm=True)
        self.dec_c3 = CombinationModule(256, 256, batch_norm=True)
        self.dec_c4 = CombinationModule(256, 256, batch_norm=True)
        self.dec_c5 = CombinationModule(256, 256, batch_norm=True)
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
        x = self.base_network(x)
        for idx, layer in enumerate(x):
            print('layer {} shape: {}'.format(idx, layer.shape))
        c5_combine = self.dec_c5(x[-1], x[-2])
        c4_combine = self.dec_c4(c5_combine, x[-3])
        c3_combine = self.dec_c3(c4_combine, x[-4])
        c2_combine = self.dec_c2(c3_combine, x[-5])


        dec_dict = {}
        for head in self.heads:
            dec_dict[head] = self.__getattr__(head)(c2_combine)
            if 'hm' in head or 'cls' in head:
                dec_dict[head] = torch.sigmoid(dec_dict[head])
        return dec_dict
class CTRBOX_EfficientNetV5(nn.Module):
    def __init__(self, heads, pretrained, down_ratio, final_kernel, head_conv):
        super().__init__()
        channels = [3, 64, 256, 512, 1024, 2048]
        assert down_ratio in [2, 4, 8, 16]
        self.l1 = int(np.log2(down_ratio))
        self.base_network = efficientnet_v2.EfficientNetV2('xl',
                        in_channels=3,
                        n_classes=256,
                        pretrained=True)
        self.adapter_layer = nn.Sequential(nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=True),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(inplace=True))
        self.dec_c2 = CombinationModule(96, 64, batch_norm=True)
        self.dec_c3 = CombinationModule(256, 96, batch_norm=True)
        self.dec_c4 = CombinationModule(640, 256, batch_norm=True)                                
        self.heads = heads

        for head in self.heads:
            classes = self.heads[head]
            if head == 'wh':
                fc = nn.Sequential(mini_inception.MiniInception(),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(channels[self.l1], head_conv, kernel_size=7, padding=3, bias=True),
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

    def fill_fc_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.base_network(x)
        for idx, layer in enumerate(x):
            print('layer {} shape: {}'.format(idx, layer
                                              .shape))
        c4_combine = self.dec_c4(x[-1], x[-2])
        print('c4_combine shape: ', c4_combine.shape)
        c3_combine = self.dec_c3(c4_combine, x[-3])
        print('c3_combine shape: ', c3_combine.shape)
        c2_combine = self.dec_c2(c3_combine, x[-4])
        print('c2_combine shape: ', c2_combine.shape)
        c2_combine = self.adapter_layer(c2_combine)
        print('c2_combine shape: ', c2_combine.shape)
        dec_dict = {}
        for head in self.heads:
            dec_dict[head] = self.__getattr__(head)(c2_combine)
            if 'hm' in head or 'cls' in head:
                dec_dict[head] = torch.sigmoid(dec_dict[head])
        return dec_dict
class CTRBOX_ifelseFilter(nn.Module):
    def __init__(self, heads, pretrained, down_ratio, final_kernel, head_conv):
        super().__init__()
        channels = [3, 64, 256, 512, 1024, 2048]
        assert down_ratio in [2, 4, 8, 16]
        self.l1 = int(np.log2(down_ratio))
        self.base_network = efficientnet_v2.EfficientNetV2('xl',
                        in_channels=3,
                        n_classes=256,
                        pretrained=True)
        self.adapter_layer = nn.Sequential(nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=True),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(inplace=True))
        self.dec_c2 = CombinationModule(96, 64, batch_norm=True)
        self.dec_c3 = CombinationModule(256, 96, batch_norm=True)
        self.dec_c4 = CombinationModule(640, 256, batch_norm=True)                                
        self.heads = heads

        for head in self.heads:
            classes = self.heads[head]
            if head == 'wh':
                fc = nn.Sequential(nn.Conv2d(channels[self.l1], ))
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

    def fill_fc_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.base_network(x)
        c4_combine = self.dec_c4(x[-1], x[-2])
        c3_combine = self.dec_c3(c4_combine, x[-3])
        c2_combine = self.dec_c2(c3_combine, x[-4])
        c2_combine = self.adapter_layer(c2_combine)
        dec_dict = {}
        for head in self.heads:
            dec_dict[head] = self.__getattr__(head)(c2_combine)
            if 'hm' in head or 'cls' in head:
                dec_dict[head] = torch.sigmoid(dec_dict[head])
        return dec_dict
class CTRBOX_101_Addition(nn.Module):
    def __init__(self, heads, pretrained, down_ratio, final_kernel, head_conv):
        super().__init__()
        channels = [3, 64, 256, 512, 1024, 2048]
        assert down_ratio in [2, 4, 8, 16]
        self.l1 = int(np.log2(down_ratio))
        self.base_network = resnet.resnet101(pretrained=pretrained)

        self.dec_c2 = CombinationModule_Addition(512, 256)
        self.dec_c3 = CombinationModule_Addition(1024, 512)
        self.dec_c4 = CombinationModule_Addition(2048, 1024)
        self.heads = heads

        for head in self.heads:
            classes = self.heads[head]
            if head == 'wh':
                fc = nn.Sequential(nn.Conv2d(channels[self.l1], head_conv, kernel_size=3, padding=1, bias=True),
                                   nn.BatchNorm2d(head_conv),   # BN not used in the paper, but would help stable training
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, classes, kernel_size=3, padding=1, bias=True))
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
        x = self.base_network(x)
        # for idx, layer in enumerate(x):
        #     print('layer {} shape: {}'.format(idx, layer
        #                                       .shape))
        # import matplotlib.pyplot as plt
        # import os
        # for idx in range(x[1].shape[1]):
        #     temp = x[1][0,idx,:,:]
        #     temp = temp.data.cpu().numpy()
        #     plt.imsave(os.path.join('dilation', '{}.png'.format(idx)), temp)


        c4_combine = self.dec_c4(x[-1], x[-2])
        c3_combine = self.dec_c3(c4_combine, x[-3])
        c2_combine = self.dec_c2(c3_combine, x[-4])
        # print('c2_combine shape: ', c2_combine.shape)

        dec_dict = {}
        for head in self.heads:
            dec_dict[head] = self.__getattr__(head)(c2_combine)
            if 'hm' in head or 'cls' in head:
                dec_dict[head] = torch.sigmoid(dec_dict[head])
        # for dec in dec_dict:
        #     print(dec, dec_dict[dec].shape)
        return dec_dict
class CTRBOX_Github_aux(nn.Module):
    def __init__(self, heads, pretrained, down_ratio, final_kernel, head_conv):
        super().__init__()
        channels = [3, 64, 256, 512, 1024, 2048]
        assert down_ratio in [2, 4, 8, 16]
        self.l1 = int(np.log2(down_ratio))
        self.base_network = resnet.resnet101(pretrained=pretrained)

        self.dec_c2 = CombinationModule(512, 256, batch_norm=True)
        self.dec_c3 = CombinationModule(1024, 512, batch_norm=True)
        self.dec_c4 = CombinationModule(2048, 1024, batch_norm=True)
        self.heads = heads
        self.aux_heads = heads
        self.adapt_aux = nn.Sequential(nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0, bias=True),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(inplace=True))

        for head in self.heads:
            classes = self.heads[head]
            if head == 'wh':
                fc = nn.Sequential(nn.Conv2d(channels[self.l1], head_conv, kernel_size=3, padding=1, bias=True),
                                   nn.BatchNorm2d(head_conv),   # BN not used in the paper, but would help stable training
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, classes, kernel_size=3, padding=1, bias=True))
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
        for head in self.aux_heads:
            classes = self.heads[head]
            if head == 'wh':
                fc = nn.Sequential(nn.Conv2d(channels[self.l1], head_conv, kernel_size=3, padding=1, bias=True),
                                   nn.BatchNorm2d(head_conv),   # BN not used in the paper, but would help stable training
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, classes, kernel_size=3, padding=1, bias=True))
            else:
                fc = nn.Sequential(nn.Conv2d(channels[self.l1], head_conv, kernel_size=3, padding=1, bias=True),
                                   nn.BatchNorm2d(head_conv),   # BN not used in the paper, but would help stable training
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, classes, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True))
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                self.fill_fc_weights(fc)

            self.__setattr__('aux_' + head, fc)

    def fill_fc_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.base_network(x)
        for idx, layer in enumerate(x):
            print('layer {} shape: {}'.format(idx, layer
                                              .shape))
        # import matplotlib.pyplot as plt
        # import os
        # for idx in range(x[1].shape[1]):
        #     temp = x[1][0,idx,:,:]
        #     temp = temp.data.cpu().numpy()
        #     plt.imsave(os.path.join('dilation', '{}.png'.format(idx)), temp)
        upscaled = F.interpolate(x[-1], size=(152, 152), mode='bilinear', align_corners=False)
        upscaled = self.adapt_aux(upscaled)

        c4_combine = self.dec_c4(x[-1], x[-2])
        c3_combine = self.dec_c3(c4_combine, x[-3])
        c2_combine = self.dec_c2(c3_combine, x[-4])
        # print('c2_combine shape: ', c2_combine.shape)

        dec_dict = {}
        for head in self.heads:
            dec_dict[head] = self.__getattr__(head)(c2_combine)
            if 'hm' in head or 'cls' in head:
                dec_dict[head] = torch.sigmoid(dec_dict[head])
        # for dec in dec_dict:
        #     print(dec, dec_dict[dec].shape)
        aux_dict = {}
        for head in self.aux_heads:
            aux_dict[head] = self.__getattr__('aux_' + head)(upscaled)
            if 'hm' in head or 'cls' in head:
                aux_dict[head] = torch.sigmoid(aux_dict[head])
        return dec_dict, aux_dict