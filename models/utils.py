import sys
sys.path.append("..")

import numpy as np
from torchvision import transforms
import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from PIL import Image
from csprites.datasets import ClassificationDataset
from torch.utils.data import DataLoader


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


def linprob_model(fn_forward, dl_linprob, device):
    R = []
    Y = []
    with torch.no_grad():
        for x, y in dl_linprob:
            x = x.to(device)
            r = fn_forward(x)
            R.append(r.detach().cpu().numpy())
            Y.append(y.cpu().numpy())
    R = np.concatenate(R)
    Y = np.concatenate(Y)
    #
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(R, Y)
    knnacc = knn.score(R, Y)
    #
    clf = LogisticRegression(random_state=0, tol=0.001, max_iter=200).fit(R, Y)
    linacc = clf.score(R, Y)
    return linacc, knnacc


def linprob_model_train_valid(fn_forward, dl_train, dl_valid, device):
    R, Y = get_representations(fn_forward, dl_train, device)
    #
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(R, Y)
    #
    clf = LogisticRegression(random_state=0, tol=0.001, max_iter=200)
    clf.fit(R, Y)
    #
    R, Y = get_representations(fn_forward, dl_valid, device)
    knnacc = knn.score(R, Y)
    linacc = clf.score(R, Y)
    return linacc, knnacc


def get_raw_csprites_dataloader(p_data, img_size, batch_size, norm_transform, num_workers=8):
    transform_representations = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        norm_transform
    ])
    # TRAIN
    ds_train = ClassificationDataset(
        p_data=p_data,
        transform=transform_representations,
        target_transform=None,
        split="train"
    )
    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False
    )
    # VALID
    ds_valid = ClassificationDataset(
        p_data=p_data,
        transform=transform_representations,
        target_transform=None,
        split="valid"
    )
    dl_valid = DataLoader(
        ds_valid,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False
    )
    return dl_train, dl_valid


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def get_representations(model_fn, dl, device, imgs=False, inverse_norm_transform=None):
    R = []
    Y = []
    X = []
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            r = model_fn(x)
            if imgs is True:
                if inverse_norm_transform is not None:
                    x = inverse_norm_transform(x.detach())
                X.append(x.cpu().numpy())
            R.append(r.detach().cpu().numpy())
            Y.append(y.cpu().numpy())
    R = np.concatenate(R)
    Y = np.concatenate(Y)
    if imgs is True:
        X = np.concatenate(X)
        X = np.transpose(X, axes=(0, 2, 3, 1))
        return R, Y, X
    else:
        return R, Y


def predict_all(R_train, Y_train, R_valid, Y_valid, target_names, p_plot=None, show=False):
    results = []
    for target_idx in range(len(target_names)):
        target = target_names[target_idx]
        if len(set(Y_train[:, target_idx])) == 1:
            print("{:>15}: acc = NA".format(target))
            results.append(np.inf)
            continue
        clf = LogisticRegression(random_state=0).fit(
            R_train, Y_train[:, target_idx])
        score = clf.score(R_valid, Y_valid[:, target_idx])
        print("{:>15}: acc = {:.2f}".format(target, score))
        results.append(score)

    fig, ax = plt.subplots(1, 1)
    ax.bar(range(len(results)), results, width=1)
    ax.set_ylim([0, 1])
    ax.set_xticks(np.arange(len(target_names)))
    ax.set_xticklabels(target_names)
    plt.title("Prediction Accurace LR on valid")
    if p_plot is not None:
        plt.savefig(p_plot)
    if show:
        plt.show()
    else:
        plt.close()


def plot_latent_by_imgs(R, X, Y, n_imgs, show=False, p_plot=None):
    topic_idcs = []
    for dim_idx in range(R.shape[1]):
        r = R[:, dim_idx]
        idcs = np.argsort(r)[-n_imgs:]
        topic_idcs.append(idcs)
    topic_idcs = np.array(topic_idcs)

    img_dim = X.shape[1]

    h, w = np.array(topic_idcs.shape) * img_dim
    img = np.zeros((h, w, 3))
    print(img.shape)
    n_rows, n_cols = topic_idcs.shape
    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            img_idx = topic_idcs[row_idx][col_idx]
            img[row_idx * img_dim: row_idx * img_dim + img_dim, col_idx *
                img_dim:col_idx * img_dim + img_dim, :] = X[img_idx]
    if show:
        plt.figure(figsize=topic_idcs.shape)
        plt.imshow(img)
        plt.show()
    if p_plot is not None:
        Image.fromarray(np.uint8(img * 255)).save(p_plot)



