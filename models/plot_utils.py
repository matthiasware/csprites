import numpy as np
import torch
from torch.distributions import Beta
import matplotlib.pyplot as plt


def imshow(img):
    npimg = img.detach().cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def plot_mat(mat, row_names=None, col_names=None, scale_factor=2, title=None, xlabel=None, ylabel=None, p_file=None):
    n_rows, n_cols = mat.shape
    fig, ax = plt.subplots(figsize=(n_cols * scale_factor,
                                    n_rows * scale_factor))
    im = ax.imshow(mat, cmap="copper")
    #
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    #

    if row_names is not None:
        assert len(row_names) == n_rows
        ax.set_yticklabels(row_names)
    if col_names is not None:
        assert len(col_names) == n_cols
        ax.set_xticklabels(col_names)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for col_idx in range(n_cols):
        for row_idx in range(n_rows):
            text = ax.text(col_idx, row_idx, "{:.2f}".format(mat[row_idx, col_idx]),
                           ha="center", va="center", color="w")
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    fig.tight_layout()
    if p_file is not None:
        plt.savefig(p_file)
    plt.show()


def simplex_plot1d(x, title=None, p_file=None, figsize=(10, 1), c=None, cmap=None):
    assert np.all(x >= 0)
    assert np.all(x <= 1)
    assert len(x.shape) == 1 or (len(x.shape) == 2 and x.shape[1] == 1)
    #
    plt.figure(figsize=figsize)
    plt.scatter(x, np.ones(len(x)), c=c, cmap=cmap)
    plt.plot([0, 1], [1, 1])
    plt.xlim([-0.05, 1.05])
    plt.yticks([])
    if title is not None:
        plt.title(title)
    if p_file is not None:
        plt.savefig(p_file)
    plt.show()


def simplex_plot2d(x, title=None, p_file=None, figsize=(8, 8), show=True, c=None, cmap=None):
    assert np.all(x.flatten() >= 0)
    assert np.all(x.flatten() <= 1)
    assert len(x.shape) == 2
    assert x.shape[1] == 3
    #
    p1 = np.array([0.0, (3.0**0.5) - 1.0])
    p2 = np.array([-1.0, -1.0])
    p3 = np.array([1.0, -1.0])
    #
    plt.figure(figsize=figsize)
    plt.plot([-1.0, 1.0, 0.0, -1.0],
             [-1.0, -1.0, (3.0**0.5) - 1.0, -1.0])
    points = []
    for t1, t2, t3 in x:
        points.append(p1 * t1 + p2 * t2 + p3 * t3)
    points = np.array(points)
    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)
    plt.scatter(points.T[0], points.T[1], edgecolors='#ffffff', alpha=0.5, c=c, cmap=cmap)
    plt.text(p1[0] - 0.1, p1[1] + 0.1, 'a1', fontsize=16)
    plt.text(p2[0] - 0.1, p2[1] - 0.15, 'a2', fontsize=16)
    plt.text(p3[0], p3[1] - 0.15, 'a3', fontsize=16)
    plt.xticks([])
    plt.yticks([])
    if title is not None:
        plt.title(title)
    if p_file is not None:
        plt.savefig(p_file)
    if show:
        plt.show()
    plt.close()


def simplex_plot(x, title=None, p_file=None, figsize=None, c=None, cmap=None):
    x = np.array(x)
    if len(x.shape) == 1 or (len(x.shape) == 2 and x.shape[1] == 1):
        if figsize is None:
            figsize = (10, 1)
        simplex_plot1d(x, title, p_file, figsize, c=c, cmap=cmap)
    elif len(x.shape) == 2 and x.shape[1] == 2:
        if figsize is None:
            figsize = (10, 1)
        simplex_plot1d(x[:, 0], title, p_file, c=c, cmap=cmap)
    elif len(x.shape) == 2 and x.shape[1] == 3:
        if figsize is None:
            figsize = (8, 8)
        simplex_plot2d(x, title, p_file, figsize, c=c, cmap=cmap)
    else:
        raise Exception("Cannot plot for input of shape {}".format(x.shape))


def plot_beta_pdf(dist=None, ab=None, title=None, p_file=None, show=True):
    xx = torch.linspace(0, 1, 200)[1:-1]
    if dist is None:
        dist = Beta(ab[0], ab[1])
    plt.plot(xx, torch.exp(dist.log_prob(xx)))
    a, b = float(dist.concentration0), float(dist.concentration1)
    if title is not None:
        plt.title("{} \n a={:.3f}, beta={:.3f}".format(
            title, a, b))
    else:
        plt.title("a={:.3f}, beta={:.3f}".format(a, b))
    if p_file is not None:
        plt.savefig(p_file)
    if show:
        plt.show()
    plt.close()


def scatter(x, show=True, p_file=None, title=None):
    if type(x) == list:
        for xi in x:
            assert len(xi.shape) == 1 or (
                len(xi.shape) == 2 and xi.shape[1] == 1)
    else:
        assert len(x.shape) == 1 or (len(x.shape) == 2 and x.shape[1] == 1)
        x = [x]
    for xi in x:
        plt.scatter(range(len(xi)), xi)
    if title is not None:
        plt.title(title)
    if p_file is not None:
        plt.savefig(p_file)
    if show:
        plt.show()
    plt.close()
