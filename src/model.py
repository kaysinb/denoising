import torch
from torch import nn
from src.settings import *


class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(channel_num, channel_num, 3, padding=1),
            nn.BatchNorm2d(channel_num),
            nn.ReLU(),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(channel_num, channel_num, 3, padding=1),
            nn.BatchNorm2d(channel_num),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x + residual
        out = self.relu(x)
        return out


def make_conv_block(input_chs, out_chs, kernel_size=3, stride=1, padding=1):
    block = nn.Sequential(
        nn.Conv2d(input_chs, out_chs, kernel_size, stride, padding),
        nn.BatchNorm2d(out_chs),
        nn.LeakyReLU(negative_slope=0.2),
    )
    return block


class UEncoder(nn.Module):
    def __init__(self, hidden_dim=4):

        super(UEncoder, self).__init__()
        self.init_conv = make_conv_block(1, hidden_dim)
        self.conv1 = make_conv_block(hidden_dim, hidden_dim * 2)
        self.conv2 = make_conv_block(hidden_dim * 2, hidden_dim * 4)
        self.conv3 = make_conv_block(hidden_dim * 4, hidden_dim * 8)

        res_block_1 = [ResBlock(hidden_dim) for i in range(NUM_RES_BLOCKS)]
        res_block_1.append(nn.MaxPool2d((2, 2)))
        self.res1 = nn.ModuleList(res_block_1)

        res_block_2 = [ResBlock(hidden_dim * 2) for i in range(NUM_RES_BLOCKS)]
        res_block_2.append(nn.MaxPool2d((2, 2)))
        self.res2 = nn.ModuleList(res_block_2)

        res_block_3 = [ResBlock(hidden_dim * 4) for i in range(NUM_RES_BLOCKS)]
        res_block_3.append(nn.MaxPool2d((4, 4)))
        self.res3 = nn.ModuleList(res_block_3)

    def forward(self, x):
        x_1 = self.init_conv(x)
        x_2 = x_1
        for i, _ in enumerate(self.res1):
            x_2 = self.res1[i](x_2)
        x_2 = self.conv1(x_2)

        x_3 = x_2
        for i, _ in enumerate(self.res2):
            x_3 = self.res2[i](x_3)
        x_3 = self.conv2(x_3)

        x_4 = x_3
        for i, _ in enumerate(self.res3):
            x_4 = self.res3[i](x_4)
        x_4 = self.conv3(x_4)

        return x_1, x_2, x_3, x_4


class UDecoder(nn.Module):
    def __init__(self, hidden_dim=4):
        super(UDecoder, self).__init__()
        self.out_conv = make_conv_block(hidden_dim * 2, 1)
        self.conv_tr_1 = nn.ConvTranspose2d(hidden_dim * 4, hidden_dim, kernel_size=(2, 2), stride=(2, 2))
        self.conv_tr_2 = nn.ConvTranspose2d(hidden_dim * 12, hidden_dim * 2, kernel_size=(2, 2), stride=(2, 2))
        self.up_3 = nn.Upsample(scale_factor=(4, 4), mode='bilinear', align_corners=True)

        self.res1 = nn.ModuleList([ResBlock(hidden_dim * 2) for i in range(NUM_RES_BLOCKS)])
        self.res2 = nn.ModuleList([ResBlock(hidden_dim * 4) for i in range(NUM_RES_BLOCKS)])
        self.res3 = nn.ModuleList([ResBlock(hidden_dim * 12) for i in range(NUM_RES_BLOCKS)])

    def forward(self, x_1, x_2, x_3, x_4):
        x = self.up_3(x_4)
        x = torch.cat([x, x_3], dim=1)
        for i, _ in enumerate(self.res1):
            x = self.res3[i](x)

        x = self.conv_tr_2(x)
        x = torch.cat([x, x_2], dim=1)
        for i, _ in enumerate(self.res1):
            x = self.res2[i](x)

        x = self.conv_tr_1(x)
        x = torch.cat([x, x_1], dim=1)
        for i, _ in enumerate(self.res1):
            x = self.res1[i](x)

        x = self.out_conv(x)
        return x


class UMelResNet(nn.Module):
    def __init__(self, hidden_dim=4):
        super(UMelResNet, self).__init__()
        self.encoder = UEncoder(hidden_dim)
        self.decoder = UDecoder(hidden_dim)
        self.classification_block = UMelResNet.make_classification_block(hidden_dim * 8, 2)
        self.soft_max = nn.Softmax(dim=1)

    @staticmethod
    def make_classification_block(input_chs, out_chs):
        block = nn.Sequential(
            nn.MaxPool2d(5),
            nn.Flatten(),
            nn.Linear(input_chs, out_chs)
        )
        return block

    def forward(self, x):
        x_1, x_2, x_3, x_4 = self.encoder(x)
        x_classification = self.classification_block(x_4)
        x_mel = self.decoder(x_1, x_2, x_3, x_4)
        return x_classification, x_mel

    def is_noisy(self, x):
        with torch.no_grad():
            x_1, x_2, x_3, x_4 = self.encoder(x)
            x_classification = self.soft_max(self.classification_block(x_4))
        out = x_classification.sum(dim=0)/x_classification.shape[0]
        return out[0].item() < out[1].item()

    def denoising(self, x):
        with torch.no_grad():
            x_1, x_2, x_3, x_4 = self.encoder(x)
            x_mel = self.decoder(x_1, x_2, x_3, x_4)
        return x_mel
