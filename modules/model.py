#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn

def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)


def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
    else:
        return nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2),
                             conv(in_channels, out_channels, kernel_size=1, stride=1, padding=0))

class UNet(nn.Module):
    def __init__(self, in_ch=3, dim=64, out_ch=2, merge_mode='concat', up_mode='transpose'):
        super(UNet, self).__init__()

        self.down1 = UNetDownBlock(in_ch, dim, 3, 1, 1)
        self.down2 = UNetDownBlock(dim, dim*2, 4, 2, 1)
        self.down3 = UNetDownBlock(dim*2, dim*4, 4, 2, 1)
        self.down4 = UNetDownBlock(dim*4, dim*8, 4, 2, 1)
        self.down5 = UNetDownBlock(dim*8, dim*8, 4, 2, 1)

        self.up1 = UNetUpBlock(dim*8, dim*8, merge_mode=merge_mode, up_mode=up_mode)
        self.up2 = UNetUpBlock(dim*8, dim*4, merge_mode=merge_mode, up_mode=up_mode)
        self.up3 = UNetUpBlock(dim*4, dim*2, merge_mode=merge_mode, up_mode=up_mode)
        self.up4 = UNetUpBlock(dim*2, dim, merge_mode=merge_mode, up_mode=up_mode)

        self.conv_final = nn.Sequential(conv(dim, out_ch, 3, 1, 1), nn.Tanh())

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.conv_final(x)
        return x

class UNet2(nn.Module):
    def __init__(self, in_ch=3, dim=64, out_ch=2, merge_mode='concat', up_mode='transpose'):
        super(UNet2, self).__init__()

        self.down1 = UNetDownBlock(in_ch, dim, 3, 1, 1)
        self.down2 = UNetDownBlock(dim, dim*2, 4, 2, 1)
        self.down3 = UNetDownBlock(dim*2, dim*4, 4, 2, 1)
        self.down4 = UNetDownBlock(dim*4, dim*8, 4, 2, 1)

        self.up2 = UNetUpBlock(dim*8, dim*4, merge_mode=merge_mode, up_mode=up_mode)
        self.up3 = UNetUpBlock(dim*4, dim*2, merge_mode=merge_mode, up_mode=up_mode)
        self.up4 = UNetUpBlock(dim*2, dim, merge_mode=merge_mode, up_mode=up_mode)

        self.conv_final = nn.Sequential(conv(dim, out_ch, 3, 1, 1), nn.Tanh())

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.conv_final(x)
        return x

class UNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(UNetDownBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv1 = conv(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = conv(self.out_channels, self.out_channels)
        self.bn2 = nn.BatchNorm2d(self.out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))

        return x


class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, merge_mode='concat', up_mode='transpose'):
        super(UNetUpBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.upconv = upconv2x2(self.in_channels, self.out_channels, mode=self.up_mode)

        if self.merge_mode == 'concat':
            self.conv1 = conv(2 * self.out_channels, self.out_channels)
        else:
            self.conv1 = conv(self.out_channels, self.out_channels)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = conv(self.out_channels, self.out_channels)
        self.bn2 = nn.BatchNorm2d(self.out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, from_up, from_down):
        from_up = self.upconv(from_up)

        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))

        return x

class CAE(nn.Module):
    def __init__(self, in_ch=3, dim=64, out_ch=3, up_mode='transpose'):
        super(CAE, self).__init__()

        self.down1 = CAEDownBlock(in_ch, dim, 3, 1, 1)
        self.down2 = CAEDownBlock(dim, dim*2, 4, 2, 1)
        self.down3 = CAEDownBlock(dim*2, dim*4, 4, 2, 1)
        self.down4 = CAEDownBlock(dim*4, dim*8, 4, 2, 1)
        self.down5 = CAEDownBlock(dim*8, dim*8, 4, 2, 1)

        self.up1 = CAEUpBlock(dim*8, dim*8, up_mode=up_mode)
        self.up2 = CAEUpBlock(dim*8, dim*4, up_mode=up_mode)
        self.up3 = CAEUpBlock(dim*4, dim*2, up_mode=up_mode)
        self.up4 = CAEUpBlock(dim*2, dim, up_mode=up_mode)

        self.conv_final = nn.Sequential(conv(dim, out_ch, 3, 1, 1), nn.Tanh())

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.conv_final(x)
        return x

class CAE2(nn.Module):
    def __init__(self, in_ch=3, dim=8, out_ch=3, up_mode='transpose'):
        super(CAE2, self).__init__()

        self.down1 = CAEDownBlock(in_ch, dim, 3, 1, 1)
        self.down2 = CAEDownBlock(dim, dim*2, 4, 2, 1)
        self.down3 = CAEDownBlock(dim*2, dim*4, 4, 2, 1)
        self.down4 = CAEDownBlock(dim*4, dim*8, 4, 2, 1)
        self.down5 = CAEDownBlock(dim*8, dim*16, 4, 2, 1)
        self.down6 = CAEDownBlock(dim*16, dim*16, 4, 2, 1)

        self.up1 = CAEUpBlock(dim*16, dim*16, up_mode=up_mode)
        self.up2 = CAEUpBlock(dim*16, dim*8, up_mode=up_mode)
        self.up3 = CAEUpBlock(dim*8, dim*4, up_mode=up_mode)
        self.up4 = CAEUpBlock(dim*4, dim*2, up_mode=up_mode)
        self.up5 = CAEUpBlock(dim*2, dim, up_mode=up_mode)

        self.conv_final = nn.Sequential(conv(dim, out_ch, 3, 1, 1), nn.Tanh())

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)
        x = self.down6(x)

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        x = self.conv_final(x)
        return x

class CAEDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(CAEDownBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv1 = conv(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = conv(self.out_channels, self.out_channels)
        self.bn2 = nn.BatchNorm2d(self.out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))

        return x

class CAEUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_mode='transpose'):
        super(CAEUpBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_mode = up_mode

        self.upconv = upconv2x2(self.in_channels, self.in_channels, mode=self.up_mode)

        self.conv1 = conv(self.in_channels, self.out_channels)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = conv(self.out_channels, self.out_channels)
        self.bn2 = nn.BatchNorm2d(self.out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.upconv(x)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))

        return x