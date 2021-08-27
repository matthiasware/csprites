import torch
import torch.nn as nn
'''
Got code from : https://github.com/MaxLikesMath/Barlow-Twins-Pytorch/blob/main/Twins/barlow.py
Implementation of Barlow Twins (https://arxiv.org/abs/2103.03230), adapted for ease of use for experiments from
https://github.com/facebookresearch/barlowtwins, with some modifications using code from 
https://github.com/lucidrains/byol-pytorch
'''


def flatten(t):
    return t.reshape(t.shape[0], -1)


class NetWrapper(nn.Module):
    # from https://github.com/lucidrains/byol-pytorch
    def __init__(self, net, layer=-2):
        super().__init__()
        self.net = net
        self.layer = layer

        self.hidden = None
        self.hook_registered = False

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _, __, output):
        self.hidden = flatten(output)

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    def get_representation(self, x):
        if self.layer == -1:
            return self.net(x)

        if not self.hook_registered:
            self._register_hook()

        _ = self.net(x)
        hidden = self.hidden
        self.hidden = None
        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden

    def forward(self, x):
        representation = self.get_representation(x)

        return representation


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwins(nn.Module):
    '''
    Adapted from https://github.com/facebookresearch/barlowtwins for arbitrary backbones, and arbitrary choice of which
    latent representation to use. Designed for models which can fit on a single GPU (though training can be parallelized
    across multiple as with any other model). Support for larger models can be done easily for individual use cases by
    by following PyTorch's model parallelism best practices.
    '''

    def __init__(self, backbone, projection_sizes, lambd, scale_factor=1):
        '''

        :param backbone: Model backbone
        :param latent_id: name (or index) of the layer to be fed to the projection MLP
        :param projection_sizes: size of the hidden layers in the projection MLP
        :param lambd: tradeoff function
        :param scale_factor: Factor to scale loss by, default is 1
        '''
        super().__init__()
        self.backbone = backbone
        self.lambd = lambd
        self.scale_factor = scale_factor
        # projector
        sizes = [backbone.dim_out] + projection_sizes
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def get_representation(self, x):
        return self.backbone(x)

    def forward(self, y1, y2):
        z1 = self.backbone(y1)
        z2 = self.backbone(y2)
        z1 = self.projector(z1)
        z2 = self.projector(z2)

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(z1.shape[0])

        # use --scale-loss to multiply the loss by a constant factor
        # see the Issues section of the readme
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = self.scale_factor * (on_diag + self.lambd * off_diag)
        return loss


class CspritesBarlowTwins(nn.Module):
    '''
    Adapted from https://github.com/facebookresearch/barlowtwins for arbitrary backbones, and arbitrary choice of which
    latent representation to use. Designed for models which can fit on a single GPU (though training can be parallelized
    across multiple as with any other model). Support for larger models can be done easily for individual use cases by
    by following PyTorch's model parallelism best practices.
    '''

    def __init__(self, backbone, projection_sizes, lambd, w_stl=0.5, w_geo=0.5, scale_factor=1):
        '''

        :param backbone: Model backbone
        :param latent_id: name (or index) of the layer to be fed to the projection MLP
        :param projection_sizes: size of the hidden layers in the projection MLP
        :param lambd: tradeoff function
        :param scale_factor: Factor to scale loss by, default is 1
        '''
        super().__init__()
        self.backbone = backbone
        self.lambd = lambd
        self.w_stl = w_stl
        self.w_geo = w_geo
        self.scale_factor = scale_factor
        # projector
        sizes = [backbone.dim_out] + projection_sizes
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        assert projection_sizes[-1] % 2 == 0

        self.d_stl = projection_sizes[-1] // 2
        self.d_geo = projection_sizes[-1] - self.d_stl
        self.bn_stl = nn.BatchNorm1d(self.d_stl, affine=False)
        self.bn_geo = nn.BatchNorm1d(self.d_geo, affine=False)

    def get_representation(self, x):
        return self.backbone(x)

    def barlow_stl_loss(self, z1, z2):
        # empirical cross-correlation matrix
        c = self.bn_stl(z1).T @ self.bn_stl(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(z1.shape[0])

        # use --scale-loss to multiply the loss by a constant factor
        # see the Issues section of the readme
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = self.scale_factor * (on_diag + self.lambd * off_diag)
        return loss

    def barlow_geo_loss(self, z1, z2):
        # empirical cross-correlation matrix
        c = self.bn_geo(z1).T @ self.bn_geo(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(z1.shape[0])

        # use --scale-loss to multiply the loss by a constant factor
        # see the Issues section of the readme
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = self.scale_factor * (on_diag + self.lambd * off_diag)
        return loss

    def forward(self, y11, y12, y21, y22):
        """
        same geo: (y11, y12)(y21, y22)
        same stl: (y11, y21), (y12, y22)
        """
        z11 = self.projector(self.backbone(y11))
        z12 = self.projector(self.backbone(y12))
        z21 = self.projector(self.backbone(y21))
        z22 = self.projector(self.backbone(y22))
        #
        z11_stl = z11[:, :self.d_stl]
        z11_geo = z11[:, self.d_stl:]
        #
        z12_stl = z12[:, :self.d_stl]
        z12_geo = z12[:, self.d_stl:]
        #
        z21_stl = z21[:, :self.d_stl]
        z21_geo = z21[:, self.d_stl:]
        #
        z22_stl = z22[:, :self.d_stl]
        z22_geo = z22[:, self.d_stl:]
        #
        # GEO LOSS
        geo_1112_loss = self.barlow_geo_loss(z11_geo, z12_geo) * self.w_geo
        geo_2122_loss = self.barlow_geo_loss(z21_geo, z22_geo) * self.w_geo

        # STL LOSS
        stl_1121_loss = self.barlow_stl_loss(z11_stl, z21_stl) * self.w_stl
        stl_1222_loss = self.barlow_stl_loss(z12_stl, z22_stl) * self.w_stl

        loss = 0.25 * (geo_1112_loss + geo_2122_loss +
                       stl_1121_loss + stl_1222_loss)
        return loss, geo_1112_loss, geo_2122_loss, stl_1121_loss, stl_1222_loss
