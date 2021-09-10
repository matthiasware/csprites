from PIL import Image, ImageOps, ImageFilter
import torch
import torchvision.transforms as transforms
import random
'''
#####
Adapted from https://github.com/facebookresearch/barlowtwins
#####
'''


class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class Transform:
    def __init__(self, transform=None, transform_prime=None):
        '''
        :param transform: Transforms to be applied to first input
        :param transform_prime: transforms to be applied to second
        '''
        if transform == None:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                            saturation=0.2, hue=0.1)],
                    p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=1.0),
                Solarization(p=0.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        if transform_prime == None:

            self.transform_prime = transforms.Compose([
                transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                            saturation=0.2, hue=0.1)],
                    p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=0.1),
                Solarization(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform_prime = transform_prime

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2


class DualGeometricTransform(torch.nn.Module):
    def __init__(self, img_size, scale, ratio, p_hflip, p_vflip):
        super().__init__()
        self.img_size = img_size
        self.scale = scale
        self.ratio = ratio
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip

    def forward(self, img1, img2):
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            img1,
            scale=self.scale,
            ratio=self.ratio)
        img1 = transforms.functional.resized_crop(
            img1, i, j, h, w, self.img_size, Image.BICUBIC)
        img2 = transforms.functional.resized_crop(
            img2, i, j, h, w, self.img_size, Image.BICUBIC)

        if torch.rand(1) < self.p_hflip:
            img1 = transforms.functional.hflip(img1)
            img2 = transforms.functional.hflip(img2)
        if torch.rand(1) < self.p_vflip:
            img1 = transforms.functional.vflip(img1)
            img2 = transforms.functional.vflip(img2)
        return img1, img2


class CSpritesTransform(torch.nn.Module):
    def __init__(self, img_size, scale, ratio, p_hflip, p_vflip, stl_transform, fin_transform):
        super().__init__()
        self.stl_transform = stl_transform
        self.geo_transform = DualGeometricTransform(
            img_size, scale, ratio, p_hflip, p_vflip)
        self.fin_transform = fin_transform

    def forward(self, x):
        x1 = self.stl_transform(x)
        x2 = self.stl_transform(x)
        #
        x11, x12 = self.geo_transform(x1, x2)
        x21, x22 = self.geo_transform(x1, x2)
        #
        if self.fin_transform is not None:
            x11 = self.fin_transform(x11)
            x12 = self.fin_transform(x12)
            x21 = self.fin_transform(x21)
            x22 = self.fin_transform(x22)
        return x11, x12, x21, x22


class CSpritesTripleTransform(torch.nn.Module):
    def __init__(self, init_transform, geo_transform, stl_transform, fin_transform):
        super().__init__()
        self.init_transform = init_transform
        self.stl_transform = stl_transform
        self.geo_transform = geo_transform
        self.fin_transform = fin_transform

    def forward(self, x):
        x = self.init_transform(x)
        #
        x_geo = self.stl_transform(x)
        x_stl = self.geo_transform(x)
        #
        if self.fin_transform is not None:
            x = self.fin_transform(x)
            x_stl = self.fin_transform(x_stl)
            x_geo = self.fin_transform(x_geo)
        return x, x_stl, x_geo