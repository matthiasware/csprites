import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class BasicConvBlock(nn.Module):
    def __init__(self, c_in, c_out, n_blocks=1, batch_norm=True, activation=True):
        super().__init__()

        chs = self.blockify(c_in, c_out, n_blocks)
        modules = []
        for idx in range(0, len(chs) - 1, 1):
            c_in = chs[idx]
            c_out = chs[idx + 1]
            modules.append(nn.Conv2d(c_in, c_out, kernel_size=3, padding=1))
            if batch_norm:
                modules.append(nn.BatchNorm2d(c_out))
            if activation:
                modules.append(nn.ReLU(inplace=True))
        self.convs = nn.Sequential(*modules)

    def forward(self, x):
        return self.convs(x)

    def blockify(self, c_in, c_out, n_blocks):
        chs = [c_in] + [c_out] * n_blocks
        return chs


class DownBlock(nn.Module):
    def __init__(self, c_in, c_out, n_blocks=1):
        super().__init__()
        self.maxpol = nn.MaxPool2d(2)
        self.convs = BasicConvBlock(
            c_in, c_out, n_blocks, batch_norm=True, activation=True)

    def forward(self, x):
        return self.convs(self.maxpol(x))


class Encoder(nn.Module):
    def __init__(self, chs_tail, chs_down, n_conv_blocks=2):
        super().__init__()
        self.tail = BasicConvBlock(
            chs_tail[0], chs_tail[1], n_conv_blocks, True, True)
        self.down_blocks = nn.ModuleList()
        for idx in range(0, len(chs_down) - 1, 1):
            self.down_blocks.append(DownBlock(
                chs_down[idx],
                chs_down[idx + 1],
                n_conv_blocks))
            # print(chs_down[idx], chs_down[idx + 1])

    def forward(self, x):
        x = self.tail(x)
        xx = [x]
        for down_block in self.down_blocks:
            x = down_block(x)
            xx.append(x)
        return xx


class UpBlock(nn.Module):
    def __init__(self, c_down, c_out, n_blocks=1):
        super().__init__()
        #
        # assert c_out * 2 == c_down
        #
        self.up = nn.ConvTranspose2d(
            c_down, c_down // 2, kernel_size=2, stride=2
        )
        self.convs = BasicConvBlock(
            c_out * 2, c_out, n_blocks, batch_norm=True, activation=True)

    def forward(self, x_down, x_side):
        x = self.up(x_down)
        x = torch.cat([x, x_side], dim=1)
        x = self.convs(x)
        return x


class Decoder(nn.Module):
    def __init__(self, chs_head, chs_up, n_conv_blocks=2, activation_last="relu"):
        super().__init__()
        self.up_blocks = nn.ModuleList()
        for idx in range(0, len(chs_up) - 1, 1):
            self.up_blocks.append(
                UpBlock(chs_up[idx], chs_up[idx + 1], n_conv_blocks)
            )
        self.head = BasicConvBlock(
            chs_head[0], chs_head[1], n_conv_blocks, True, False)
        if activation_last == "relu":
            self.last = torch.nn.ReLU(inplace=True)
        elif activation_last == "sigmoid":
            self.last = torch.nn.Sigmoid()
        else:
            raise NotImplementedError("Activation: {}".format(activation_last))

    def forward(self, xx):
        x = xx[0]
        for idx in range(len(self.up_blocks)):
            x_side = xx[idx + 1]
            x = self.up_blocks[idx](x, x_side)
        x = self.head(x)
        x = self.last(x)
        return x


class UNet(nn.Module):
    def __init__(self, chs_tail, chs_down, chs_up, chs_head, n_conv_blocks, activation_last="relu"):
        super().__init__()
        self.encoder = Encoder(chs_tail, chs_down, n_conv_blocks)
        self.decoder = Decoder(chs_head, chs_up, n_conv_blocks, activation_last)

    def forward(self, x):
        xx = self.encoder(x)
        return self.decoder(xx[::-1])
