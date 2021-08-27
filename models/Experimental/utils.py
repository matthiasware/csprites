import random
#
import numpy as np
import matplotlib.pyplot as plt


def plot_mat(mat, row_names=None, col_names=None, scale_factor=2, title=None, xlabel=None, ylabel=None):
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
    plt.show()


def dir_sample_simple(alpha):
    gammas = [random.gammavariate(a, 1) for a in alpha]
    norm = sum(gammas)
    return np.array([g / norm for g in gammas])


def dir_sample(alpha, n=1):
    return np.array([dir_sample_simple(alpha) for _ in range(n)])


def dir_plot2d(x=None, alphas=None, n=100):
    if x is None:
        assert len(alphas) == 2
        assert n >= 1
        x = dir_sample(alphas, n)
    else:
        x = np.array(x)
        assert x.shape[1] == 2
    plt.figure(figsize=(5, 1))
    plt.plot([0, 1], [1, 1])
    x = x[:, 0]
    plt.scatter(x, np.ones(len(x)), edgecolors='red', alpha=0.5)
    #
    plt.xticks([])
    plt.yticks([])
    #
    if alphas is not None:
        plt.title("a0={:.3f} a1={:.3f}".format(alphas[0], alphas[1]))
    plt.show()


def dir_plot3d(x=None, alphas=None, n=100):
    p1 = np.array([0.0, (3.0**0.5) - 1.0])
    p2 = np.array([-1.0, -1.0])
    p3 = np.array([1.0, -1.0])
    plt.figure(figsize=(5, 5))
    plt.plot([-1.0, 1.0, 0.0, -1.0],
             [-1.0, -1.0, (3.0**0.5) - 1.0, -1.0])
    if x is None:
        assert alphas is not None
        assert len(alphas) == 3
        assert n >= 1
        x = dir_sample(alphas, n)
    else:
        x = np.array(x)
    assert x.shape[1] == 3
    points = []
    for t1, t2, t3 in x:
        points.append(p1 * t1 + p2 * t2 + p3 * t3)
    points = np.array(points)
    if alphas is not None:
        plt.title('a = [{0}, {1}, {2}] n={3}'.format(
            alphas[0], alphas[1], alphas[2], n), fontsize=16)
    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)
    plt.scatter(points.T[0], points.T[1], edgecolors='#ffffff', alpha=0.5)
    plt.text(p1[0] - 0.1, p1[1] + 0.1, 'P1', fontsize=16)
    plt.text(p2[0] - 0.1, p2[1] - 0.15, 'P2', fontsize=16)
    plt.text(p3[0], p3[1] - 0.15, 'P3', fontsize=16)
    plt.xticks([])
    plt.yticks([])
    plt.show()


def beta_plot1d(x, figsize=(10, 1)):
    x = np.array(x).squeeze()
    assert len(x.shape) == 1
    plt.figure(figsize=figsize)
    plt.scatter(x, np.ones(len(x)))
    plt.plot([0, 1], [1, 1])
    plt.xlim([-0.2, 1.2])
    plt.show()


def plot_beta_pdf(dist, title=None):
    xx = torch.linspace(0, 1, 200)[1:-1]
    plt.plot(xx, torch.exp(dist.log_prob(xx)))
    a, b = float(dist.concentration0), float(dist.concentration1)
    if title is not None:
        plt.title("{} \n a={:.3f}, beta={:.3f}".format(
            title, a, b))
    else:
        plt.title("a={:.3f}, beta={:.3f}".format(a, b))
    plt.show()
