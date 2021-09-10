import torch
import torchvision
import numpy as np
from vit_pytorch import ViT
from fcn import FCN


def get_activation(activation):
    if activation == "sigmoid":
        return torch.nn.Sigmoid()
    elif activation == "tanh":
        return torch.nn.Tanh()
    elif activation == "relu":
        return torch.nn.ReLU(inplace=True)
    elif activation == "softmax":
        return torch.nn.Softmax()
    else:
        raise NotImplementedError("Activation '{}'".format(activation))


def get_model_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def get_projection_head(dim_in, dim_out):
    model = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=False),
        torch.nn.Linear(dim_in, dim_out, bias=True)
    )
    return model


def get_mobilenet_v2_backbone(**kwargs):
    model = torchvision.models.mobilenet_v2(**kwargs)
    #
    features_dim_out = model.classifier[1].in_features
    #
    model = torch.nn.Sequential(
        model.features,
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(1),
    )
    model.dim_out = features_dim_out
    return model


def get_mobilenet_v3_large_backbone(**kwargs):
    model = torchvision.models.mobilenet_v3_large(**kwargs)
    #
    features_dim_out = model.classifier[0].in_features
    #
    model = torch.nn.Sequential(
        model.features,
        model.avgpool,
        torch.nn.Flatten(1),
    )
    model.dim_out = features_dim_out
    return model


def get_mobilenet_v3_small_backbone(**kwargs):
    model = torchvision.models.mobilenet_v3_small(**kwargs)
    #
    features_dim_out = model.classifier[0].in_features
    #
    model = torch.nn.Sequential(
        model.features,
        model.avgpool,
        torch.nn.Flatten(1),
    )
    model.dim_out = features_dim_out
    return model


def get_mnasnet_05_backbone(**kwards):
    model = torchvision.models.mnasnet0_5()
    #
    features_dim_out = model.classifier[1].in_features
    #
    model = torch.nn.Sequential(
        model.layers,
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(1),
    )
    model.dim_out = features_dim_out
    return model


def get_mnasnet_10_backbone(**kwards):
    model = torchvision.models.mnasnet1_0()
    #
    features_dim_out = model.classifier[1].in_features
    #
    model = torch.nn.Sequential(
        model.layers,
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(1),
    )
    model.dim_out = features_dim_out
    return model


def get_mnasnet_13_backbone(**kwards):
    model = torchvision.models.mnasnet1_3()
    #
    features_dim_out = model.classifier[1].in_features
    #
    model = torch.nn.Sequential(
        model.layers,
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(1),
    )
    model.dim_out = features_dim_out
    return model


def get_densenet_121_backbone(**kwargs):
    model = torchvision.models.densenet121(**kwargs)
    #
    features_dim_out = model.classifier.in_features

    model = torch.nn.Sequential(
        model.features,
        torch.nn.ReLU(inplace=True),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(1),
    )
    model.dim_out = features_dim_out
    return model


def get_densenet_161_backbone(**kwargs):
    model = torchvision.models.densenet161(**kwargs)
    #
    features_dim_out = model.classifier.in_features

    model = torch.nn.Sequential(
        model.features,
        torch.nn.ReLU(inplace=True),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(1),
    )
    model.dim_out = features_dim_out
    return model


def get_densenet_169_backbone(**kwargs):
    model = torchvision.models.densenet169(**kwargs)
    #
    features_dim_out = model.classifier.in_features

    model = torch.nn.Sequential(
        model.features,
        torch.nn.ReLU(inplace=True),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(1),
    )
    model.dim_out = features_dim_out
    return model


def get_densenet_201_backbone(**kwargs):
    model = torchvision.models.densenet201(**kwargs)
    #
    features_dim_out = model.classifier.in_features

    model = torch.nn.Sequential(
        model.features,
        torch.nn.ReLU(inplace=True),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(1),
    )
    model.dim_out = features_dim_out
    return model


def get_resnet_backbone(n_layers, add_linear=False, activation=None, **kwargs):
    class_map = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152
    }
    model = class_map[n_layers](**kwargs)
    model.dim_out = model.fc.in_features
    model.fc = torch.nn.Identity()
    if add_linear is True:
        head = torch.nn.Sequential(
            #torch.nn.Linear(model.dim_out, model.dim_out, bias=False),
            get_activation(activation)
        )
        model.fc = head
    return model


def get_wide_resnet_backbone(n_layers, **kwargs):
    class_map = {
        50: torchvision.models.wide_resnet50_2,
        101: torchvision.models.wide_resnet101_2,
    }
    model = class_map[n_layers](**kwargs)
    model.dim_out = model.fc.in_features
    model.fc = torch.nn.Identity()
    return model


def get_wide_resneXt_backbone(n_layers, **kwargs):
    class_map = {
        50: torchvision.models.resnext50_32x4d,
        101: torchvision.models.resnext101_32x8d,
    }
    model = class_map[n_layers](**kwargs)
    model.dim_out = model.fc.in_features
    model.fc = torch.nn.Identity()
    return model


def get_ViT(**kwargs):
    model = ViT(**kwargs)
    model.dim_out = model.mlp_head[1].out_features
    return model


def get_resnet9(pretrained=False, progress=False, **kwargs):
    import torchvision.models.resnet as ptr
    model = ptr._resnet('resnet9', ptr.BasicBlock, [1, 1, 1, 1], pretrained, progress, **kwargs)
    model.fc = torch.nn.Identity()
    model.dim_out = 512
    return model


def get_fcn(**kwargs):
    model = FCN(**kwargs)
    model.fc = torch.nn.Identity()
    return model


def get_fcn32i223o128(ch_last, **kwargs):
    # ~ 500k
    planes = [32, 64, 64, 128, ch_last]
    blocks = [1, 2, 2, 3, 1]
    model = FCN(planes=planes, blocks=blocks, **kwargs)
    model.fc = torch.nn.Identity()
    return model


def get_fcn16i223o64(ch_last, **kwargs):
    # ~ 188k
    planes = [16, 32, 32, 64, ch_last]
    blocks = [1, 2, 2, 3, 1]
    model = FCN(planes=planes, blocks=blocks, **kwargs)
    model.fc = torch.nn.Identity()
    return model


def get_fcn8i223o32(ch_last, **kwargs):
    # ~ 50k
    planes = [8, 16, 32, 32, ch_last]
    blocks = [1, 2, 2, 3, 1]
    model = FCN(planes=planes, blocks=blocks, **kwargs)
    model.fc = torch.nn.Identity()
    return model


ALL_BACKBONES = [
    "FCN",
    "FCN32i223o128",
    "FCN16i223o64",
    "FCN8i223o32",
    "MobileNet-v2",
    "MobileNet-v3-Small",
    "MobileNet-v3-Large",
    "MNASNet0.5",
    "MNASNet1.0",
    "MNASNet1.3",
    "Densenet-121",
    "Densenet-161",
    "Densenet-169",
    "Densenet-201",
    "ResNet-9",
    "ResNet-18",
    "ResNet-34",
    "ResNet-50",
    "ResNet-101",
    "ResNet-152",
    "Wide-ResNet-50-2",
    "Wide-ResNet-101-2",
    "ResNeXt-50-32x4d",
    "ResNeXt-101-32x8d",
    "ViT"
]


def get_backbone(backbone: str, **kwargs):
    if backbone == "MobileNet-v2":
        model = get_mobilenet_v2_backbone(**kwargs)
    elif backbone == "MobileNet-v3-Large":
        model = get_mobilenet_v3_large_backbone(**kwargs)
    elif backbone == "MobileNet-v3-Small":
        model = get_mobilenet_v3_small_backbone(**kwargs)
    elif backbone == "MNASNet0.5":
        model = get_mnasnet_05_backbone(**kwargs)
    elif backbone == "MNASNet1.0":
        model = get_mnasnet_10_backbone(**kwargs)
    elif backbone == "MNASNet1.3":
        model = get_mnasnet_13_backbone(**kwargs)
    elif backbone == "Densenet-121":
        model = get_densenet_121_backbone(**kwargs)
    elif backbone == "Densenet-169":
        model = get_densenet_169_backbone(**kwargs)
    elif backbone == "Densenet-161":
        model = get_densenet_161_backbone(**kwargs)
    elif backbone == "Densenet-201":
        model = get_densenet_201_backbone(**kwargs)
    elif backbone == "ResNet-9":
        model = get_resnet9(**kwargs)
    elif backbone == "ResNet-18":
        model = get_resnet_backbone(18, **kwargs)
    elif backbone == "ResNet-34":
        model = get_resnet_backbone(34, **kwargs)
    elif backbone == "ResNet-50":
        model = get_resnet_backbone(50, **kwargs)
    elif backbone == "ResNet-101":
        model = get_resnet_backbone(101, **kwargs)
    elif backbone == "ResNet-152":
        model = get_resnet_backbone(152, **kwargs)
    elif backbone == "ResNeXt-50-32x4d":
        model = get_wide_resneXt_backbone(50, **kwargs)
    elif backbone == "ResNeXt-101-32x8d":
        model = get_wide_resneXt_backbone(101, **kwargs)
    elif backbone == "Wide-ResNet-50-2":
        model = get_wide_resnet_backbone(50, **kwargs)
    elif backbone == "Wide-ResNet-101-2":
        model = get_wide_resnet_backbone(101, **kwargs)
    elif backbone == "ViT":
        model = get_ViT(**kwargs)
    elif backbone == "FCN":
        model = get_fcn(**kwargs)
    elif backbone == "FCN32i223o128":
        model = get_fcn32i223o128(**kwargs)
    elif backbone == "FCN16i223o64":
        model = get_fcn16i223o64(**kwargs)
    elif backbone == "FCN8i223o32":
        model = get_fcn8i223o32(**kwargs)
    else:
        raise NotImplementedError(f"backbone: {backbone}")
    return model