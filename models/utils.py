import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms


def imshow(img):
    npimg = img.detach().cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def normalize_transform(means, stds):
    return transforms.Normalize(
        mean=means,
        std=stds)


def inverse_normalize_transform(means, stds):
    return transforms.Normalize(
        mean=-1 * np.array(means) / np.array(stds),
        std=1 / np.array(stds))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
