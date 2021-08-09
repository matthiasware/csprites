import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from pathlib import Path
from .angles import *
from torchvision import transforms
import math
import shutil
import os


def pad_mask(mask, pad_to):
    #
    assert mask.shape[0] == mask.shape[1]
    assert pad_to % 2 == 1
    assert mask.shape[0] % 2 == 1
    #
    pad = pad_to - mask.shape[0]
    assert pad % 2 == 0
    pad = pad // 2
    mask = np.pad(mask, [(pad, pad), (pad, pad)], 'constant')
    return mask


def animate(img, file, interval):
    frames = []  # for storing the generated images
    fig = plt.figure()
    plt.axis('off')
    for i in range(len(img)):
        frames.append([plt.imshow(img[i], cmap=cm.Greys_r, animated=True)])

    ani = animation.ArtistAnimation(fig, frames, interval=interval, blit=True,
                                    repeat_delay=1000)
    plt.tight_layout()
    ani.save(file)
    plt.close()


def plot_mask_series(masks):
    fig, axes = plt.subplots(1, len(masks), figsize=(len(masks) * 4, 4))
    axes = axes.flatten()
    for idx in range(len(masks)):
        ax = axes[idx]
        ax.imshow(masks[idx], cmap="gray")
        ax.set_axis_off()
    plt.show()


def shape_sizes_to_html(shape_sizes, angles, max_mask_size, p_data, interval=200):
    p_data = Path(p_data)
    p_data.mkdir(exist_ok=True)

    shapes = list(shape_sizes.keys())
    mask_paths = []
    for shape in shapes:
        sizes = shape_sizes[shape]
        sizes = [sizes[0], sizes[-1]]
        size_paths = []
        for size in sizes:
            masks = []
            for angle in angles:
                mask = shape.create(size)
                mask = apply_angle(mask, angle)
                mask = pad_mask(mask, max_mask_size)
                masks.append(mask)

            p_gif = p_data / "{}_s{}.gif".format(shape.name, size)
            size_paths.append(p_gif)
            animate(masks, p_gif, interval)

        mask_paths.append(size_paths)

    n_cols = len(mask_paths)
    n_rows = len(mask_paths[0])

    html_str = ""
    for row_idx in range(n_rows):
        row_str = '''<div style="display: flex; justify-content: row;">'''
        for col_idx in range(n_cols):
            p_gif = mask_paths[col_idx][row_idx]
            row_str += '''<img src="{}">'''.format(p_gif)
        row_str += '''</div>'''
        html_str += row_str
    return html_str


def masks_to_html_animation(masks, p_gif="tmp/tmp.gif", interval=200):
    animate(masks, p_gif, interval)
    html_str = '''<div style="display: flex; justify-content: row;">
    <img src="{}"></div>'''.format(p_gif)
    return html_str


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


class MeanStdTracker:
    """
        Approximation of the mean and std of the dataset
        Overflow save and memory friendly ;)
        But slow as fuck
        see:
         - https://www.thoughtco.com/sum-of-squares-formula-shortcut-3126266
         - https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/03/08/image-mean-std.html
    """

    def __init__(self):
        self.psum = [0, 0, 0]
        self.psum_rem = [0., 0., 0.]
        self.psum_sq = [0, 0, 0]
        self.psum_sq_rem = [0., 0., 0.]
        self.count = 0

    def add(self, x):
        x_sum = x.sum(axis=(0, 1))
        x_sum_sq = (x**2).sum(axis=(0, 1))

        x_sum_fl = [math.floor(i) for i in x_sum]
        x_sum_sq_fl = [math.floor(i) for i in x_sum_sq]

        x_sum_diff = [x_sum[i] - x_sum_fl[i] for i in range(3)]
        x_sum_sq_diff = [x_sum_sq[i] - x_sum_sq_fl[i] for i in range(3)]

        self.psum[0] += x_sum_fl[0]
        self.psum[1] += x_sum_fl[1]
        self.psum[2] += x_sum_fl[2]

        self.psum_rem[0] += x_sum_diff[0]
        self.psum_rem[1] += x_sum_diff[1]
        self.psum_rem[2] += x_sum_diff[2]

        self.psum_sq[0] += x_sum_sq_fl[0]
        self.psum_sq[1] += x_sum_sq_fl[1]
        self.psum_sq[2] += x_sum_sq_fl[2]

        self.psum_sq_rem[0] += x_sum_sq_diff[0]
        self.psum_sq_rem[1] += x_sum_sq_diff[1]
        self.psum_sq_rem[2] += x_sum_sq_diff[2]

        self.count += x.shape[0] * x.shape[1]

    def get(self):
        psum = self.psum
        psum_sq = self.psum_sq
        psum_rem = self.psum_rem
        psum_sq_rem = self.psum_sq_rem
        count = self.count

        psum_rem = [math.floor(v) for v in psum_rem]
        psum_sq_rem = [math.floor(v) for v in psum_sq_rem]

        psum = [psum[idx] + psum_rem[idx] for idx in range(3)]
        psum_sq = [psum_sq[idx] + psum_sq_rem[idx] for idx in range(3)]

        mean = [v / count for v in psum]
        var = [psum_sq[idx] / count - mean[idx]**2 for idx in range(3)]
        std = [math.sqrt(v) for v in var]
        return mean, std


def copy_and_overwrite_dir(from_path, to_path):
    if os.path.exists(to_path):
        shutil.rmtree(to_path)
    shutil.copytree(from_path, to_path)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_max_dim(masks):
    heights, widths = zip(*[mask.shape for mask in masks])
    assert heights == widths
    d_max = max(heights)
    return d_max
