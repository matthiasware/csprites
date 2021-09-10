import numpy as np
from torchvision import transforms
import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


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


# def get_representations(model_fn, dl_linprob, device):
#     R = []
#     Y = []
#     with torch.no_grad():
#         for x, y in dl_linprob:
#             x = x.to(device)
#             r = model_fn(x)
#             R.append(r.detach().cpu().numpy())
#             Y.append(y.cpu().numpy())
#     R = np.concatenate(R)
#     Y = np.concatenate(Y)
#     return R, Y

# def get_representations_and_imgs(model_fn, dl_linprob, device):
#     R = []
#     Y = []
#     with torch.no_grad():
#         for x, y in dl_linprob:
#             x = x.to(device)
#             r = model_fn(x)
#             R.append(r.detach().cpu().numpy())
#             Y.append(y.cpu().numpy())
#     R = np.concatenate(R)
#     Y = np.concatenate(Y)
#     return R, Y


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
        X = np.transpose(X, axes=(0,2,3,1))
        return R, Y, X
    else:
        return R,Y