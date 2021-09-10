import torch.nn as nn

class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1, kernel_size=3, padding=1):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_planes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out


class FCN(nn.Module):
    def __init__(self, dim_in=3, planes=[16, 32, 64], blocks=[1, 1, 1], n_class=10):
        super().__init__()
        self.planes = planes
        self.blocks = blocks
        self.n_class = n_class
        self.dim_in = 3
        self.dim_out = planes[-1]

        # INTRO
        layers = [BasicBlock(dim_in, planes[0], stride=1,
                             kernel_size=7, padding=3)]
        for _ in range(blocks[0] - 1):
            layers.append(BasicBlock(
                planes[0], planes[0], stride=1, kernel_size=3))

        # DOWNSAMPLING LAYERS
        for idx in range(0, len(planes) - 1):
            #
            n_blocks = blocks[idx + 1]
            in_planes = planes[idx]
            out_planes = planes[idx + 1]
            # DOWNSAMLING
            layers.append(BasicBlock(in_planes, out_planes, 2))
            for _ in range(n_blocks - 1):
                layers.append(BasicBlock(out_planes, out_planes, 1))
        self.layers = nn.Sequential(*layers)

        # OUTRO
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # self.fc1 = nn.Linear(self.dim_out, self.dim_out)
        # self.bn1 = nn.BatchNorm1d(self.dim_out)
        # self.relu1 = nn.ReLU(inplace=True)
        self.fc = nn.Linear(self.dim_out, self.n_class)

    def forward(self, x):
        out = self.layers(x)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        # out = self.fc1(out)
        out = self.fc(out)
        return out
