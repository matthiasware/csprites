import torch.nn as nn


class BetaTwins(nn.Module):
    def __init__(self, backbone, beta_projector, barlow_projector):
        super().__init__()
        self.backbone = backbone
        self.beta_projector = beta_projector
        self.barlow_projector = barlow_projector
        #
        self.bn = nn.BatchNorm1d(self.barlow_projector.dim_out, affine=False)
        self.beta_bn = nn.BatchNorm1d(self.beta_projector.dim_out, affine=False)
        #
        self.bn_geo = nn.BatchNorm1d(self.barlow_projector.dim_out, affine=False)
        self.bn_stl = nn.BatchNorm1d(self.barlow_projector.dim_out, affine=False)

    def backbone_proj(self, x):
        return self.backbone(x)

    def beta_proj(self, x):
        return self.beta_projector(self.backbone(x))

    def barlow_proj(self, x):
        return self.barlow_projector(self.beta_proj(x))

    def forward(self, x):
        return self.beta_proj(x)


def get_activation(act):
    if act == "Sigmoid":
        return nn.Sigmoid()
    elif act == "Softmax":
        return nn.Softmax()
    elif act == "ReLU":
        return nn.ReLU(inplace=True)
    else:
        raise NotImplementedError("Activation {}".format(act))


def get_projector_layers(sizes, activation_last=None, batchnorm_last=False):
    assert len(sizes) > 1
    layers = []
    for i in range(len(sizes) - 2):
        layers.extend([
            nn.Linear(sizes[i], sizes[i + 1], bias=False),
            nn.BatchNorm1d(sizes[i + 1]),
            nn.ReLU(inplace=True),
        ])
    layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
    if batchnorm_last is True:
        layers.append(nn.BatchNorm1d(sizes[-1]))
    if activation_last is not None:
        layers.append(get_activation(activation_last))
    return layers


def get_projector(planes_in: int, sizes: list, activation_last: str = None):
    sizes = [planes_in] + sizes
    if len(sizes) > 1:
        layers = get_projector_layers(sizes, activation_last=activation_last)
    elif len(sizes) == 1 and activation_last is not None:
        layers = [get_activation(activation_last)]
    else:
        layers = [nn.Identity()]
    projector = nn.Sequential(*layers)
    projector.dim_out = sizes[-1]
    return projector
