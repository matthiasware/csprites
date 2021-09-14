import sys
sys.path.append("..")
sys.path.append("../..")

# Python
from pathlib import Path
import os
import warnings
import math
import datetime
import time
warnings.filterwarnings('ignore')

# TORCH
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.resnet import resnet18

# MISC
from tqdm import tqdm
import pprint
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from dotted_dict import DottedDict
import pickle

from BTwins.utils import calc_lambda
from BTwins.barlow import *
from BTwins.transform_utils import *
from csprites.datasets import ClassificationDataset
import utils
from backbone import get_backbone
from optimizer import get_optimizer
from Beta.models import get_projector


def main(config):
    pprint.pprint(config)
    #
    # TORCH SETTINGS
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices
    device = torch.device(config.device)

    p_ds_config = Path(config.p_data) / "config.pkl"

    with open(p_ds_config, "rb") as file:
        ds_config = pickle.load(file)

    target_variable = config.target_variable
    target_idx = [idx for idx, target in enumerate(ds_config["classes"]) if target == target_variable][0]
    n_classes = ds_config["n_classes"][target_variable]

    norm_transform = utils.normalize_transform(
        ds_config["means"],
        ds_config["stds"])
    inverse_norm_transform = utils.inverse_normalize_transform(
        ds_config["means"],
        ds_config["stds"]
    )
    target_transform = lambda x: x[target_idx]
    #
    init_transform = lambda x: x
    stl_transform = transforms.Compose([
                    transforms.RandomApply(
                        [transforms.ColorJitter(brightness=(0.8, 1.2),
                                                contrast=(0.6, 1.8),
                                                saturation=(0.7, 1.3),
                                                hue=(-0.5, 0.5))],
                        p=1.0
                    ),
                    transforms.RandomGrayscale(p=0.1),
                    GaussianBlur(p=0.3),
                    Solarization(p=0)
    ])

    geo_transform = transforms.Compose([
        transforms.RandomResizedCrop(ds_config["img_size"],
                                     scale=(0.6, 1.5),
                                     ratio=(1, 1),
                                     interpolation=Image.BICUBIC),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5)
    ])

    fin_transform = transforms.Compose([
                    transforms.ToTensor(),
                    norm_transform
                ])

    train_transform = CSpritesTripleTransform(
        init_transform = init_transform,
        stl_transform=stl_transform,
        geo_transform=geo_transform,
        fin_transform=fin_transform
    )

    transform_linprob = transforms.Compose([
                    transforms.Resize(ds_config["img_size"]),
                    transforms.ToTensor(),
                    norm_transform
                ])

    # TRAIN
    ds_train = ClassificationDataset(
        p_data = config.p_data,
        transform=train_transform,
        target_transform=target_transform,
        split="train"
    )
    dl_train = DataLoader(
        ds_train,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=False,
        drop_last=True
    )
    # LINPROB
    ds_linprob_train = ClassificationDataset(
        p_data = config.p_data,
        transform=transform_linprob,
        target_transform=target_transform,
        split="train"
    )
    dl_linprob_train = DataLoader(
        ds_linprob_train,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers = config.num_workers,
        pin_memory=False
    )
    ds_linprob_valid = ClassificationDataset(
        p_data = config.p_data,
        transform=transform_linprob,
        target_transform=target_transform,
        split="valid"
    )
    dl_linprob_valid = DataLoader(
        ds_linprob_valid,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers = config.num_workers,
        pin_memory=False
    )


    class BarlowTwins(nn.Module):
        def __init__(self, backbone, projector, dim_cnt, dim_stl, dim_geo):
            super().__init__()
            self.backbone = backbone
            self.projector = projector
            self.dim_cnt = dim_cnt
            self.dim_stl = dim_stl
            self.dim_geo = dim_geo
            
            self.bn = nn.BatchNorm1d(projector.dim_out, affine=False)
            #self.bn_cnt = nn.BatchNorm1d(dim_cnt, affine=False)
            #self.bn_geo = nn.BatchNorm1d(dim_geo, affine=False)
            #self.bn_stl = nn.BatchNorm1d(dim_stl, affine=False)

        def get_representation(self, x):
            return self.backbone(x)

        def forward(self, x):
            return self.projector(self.backbone(x))


    # backbone
    backbone = get_backbone(config.backbone, **config.backbone_args)

    # barlow projector
    barlow_projector = get_projector(planes_in=backbone.dim_out, sizes=config.projector)

    overlap_cnt = config["cnt_overlap"]
    ratio_stl_geo = config["ratio_stl_geo"]


    dim_cnt = int(barlow_projector.dim_out * overlap_cnt)
    dim_stl_geo = barlow_projector.dim_out - dim_cnt
    dim_stl = int(ratio_stl_geo * dim_stl_geo)
    dim_geo = dim_stl_geo - dim_stl
    #
    print("dim_stl: {} dim_geo: {} dim_overlap: {}".format(dim_stl, dim_geo, dim_cnt))
    #
    model = BarlowTwins(backbone, barlow_projector, dim_cnt=dim_cnt, dim_stl=dim_stl, dim_geo=dim_geo)
    #
    if torch.cuda.device_count() > 1 and device != "cpu":
        print("Using {} gpus!".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
        model.backbone = model.module.backbone
    elif device != "cpu":
        print("Using 1 GPU!")
    else:
        print("Using CPU!")
    model = model.to(device)
    print(model)


    w_off_stl = calc_lambda(dim_cnt + dim_stl)
    w_off_geo = calc_lambda(dim_cnt + dim_geo)

    optimizer = get_optimizer(config.optimizer, model.parameters(), config.optimizer_args)


    stats = {
        'train': {
            'loss': [],
            'epoch': [],
        },
        'linprob': {
            'linacc': [],
            'knnacc': [],
            'epoch': [],
        }
    }
    stats = DottedDict(stats)
    #
    p_experiment = Path(config.p_experiment)
    p_experiment.mkdir(exist_ok=True, parents=True)
    p_ckpts = p_experiment / config.p_ckpts
    p_ckpts.mkdir(exist_ok=True)



    def cc_loss(z1, z2):
        c = z1.T @ z2
        c.div_(z1.shape[0])
            
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        return on_diag, off_diag

    def feature_split(z, dim, overlap=None):
        if overlap:
            z1 = z[:, : dim + overlap]
            z2 = z[:, dim:]
        else:
            z1 = z[:, :dim]
            z2 = z[:, dim:]
        return z1, z2

    global_step = 0
    for epoch_idx in range(1, config.num_epochs + 1, 1):
        ################
        # TRAIN
        ################
        model.train()
        epoch_step = 0
        epoch_loss = 0
       
        desc = "[{:3}/{:3}]".format(epoch_idx, config.num_epochs)
        pbar = tqdm(dl_train, bar_format= desc + '{bar:10}{n_fmt}/{total_fmt}{postfix}')
        #
        for (x_ori, x_stl, x_geo), _ in pbar:
            x_ori = x_ori.to(device)
            x_stl = x_stl.to(device)
            x_geo = x_geo.to(device)

            for param in model.parameters():
                param.grad = None
            
            # PROJECT
            z_stl = model(x_stl)
            z_geo = model(x_geo)
            z_ori = model(x_ori)
            
            z_stl = model.bn(z_stl)
            z_geo = model.bn(z_geo)
            z_ori = model.bn(z_ori)
            
            # split
            z_ori_stl, z_ori_geo = feature_split(z_ori, dim_stl, overlap=dim_cnt)
            z_stl_stl, _ = feature_split(z_stl, dim_stl, overlap=dim_cnt)
            _, z_geo_geo = feature_split(z_geo, dim_stl, overlap=dim_cnt)
            
            # CC LOSS
            on_diag_stl, off_diag_stl = cc_loss(z_ori_stl, z_stl_stl)
            on_diag_geo, off_diag_geo = cc_loss(z_ori_geo, z_geo_geo)
            
            
            loss_stl = on_diag_stl + w_off_stl * off_diag_stl
            loss_geo = on_diag_geo + w_off_geo * off_diag_geo
            
            loss = loss_stl + loss_geo

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_step += 1
            global_step += 1
            #
            pbar.set_postfix({
                'L': loss.item(),
                'on_s': on_diag_stl.item(),
                'on_g': on_diag_geo.item(),
                'od_s': off_diag_stl.item(),
                'od_g': off_diag_geo.item()
            })

        stats.train.loss.append(epoch_loss / epoch_step)
        stats.train.epoch.append(epoch_idx)

        ################
        # Linprob
        ################
        if epoch_idx % config.freqs.linprob == 0 or epoch_idx == config.num_epochs:
            model.eval()
            linacc, knnacc = utils.linprob_model(model.backbone, dl_linprob_valid, device)
            print("    Linprob Eval @LR: {:.2f} @KNN: {:.2f}".format(linacc, knnacc))
            stats.linprob.epoch.append(epoch_idx)
            stats.linprob.knnacc.append(knnacc)
            stats.linprob.linacc.append(linacc)
            model.train()
        # Checkpoint
        if epoch_idx % config.freqs.ckpt == 0 or epoch_idx == config.num_epochs:
            print("save model!")
            if torch.cuda.device_count() > 1 and device != "cpu":
                torch.save(model.module.state_dict(), p_ckpts / config.p_model.format(epoch_idx))
            else:
                torch.save(model.state_dict(), p_ckpts / config.p_model.format(epoch_idx))


    # plot losses
    plt.plot(stats.train.epoch, stats.train.loss, label="train")
    plt.legend()
    plt.savefig(p_experiment / "barlow_loss.png")
    plt.close()


    # plot linprob acc
    plt.plot(stats.linprob.epoch, stats.linprob.knnacc, label="knn")
    plt.plot(stats.linprob.epoch, stats.linprob.linacc, label="lin")
    plt.yticks([0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1])
    plt.legend()
    plt.savefig(p_experiment / "linprob_acc.png")
    plt.close()


    with open(p_experiment / config.p_config, "wb") as file:
        pickle.dump(config, file)
    with open(p_experiment / config.p_stats, "wb") as file:
        pickle.dump(stats, file)


    p_R_train = p_experiment / config["p_R_train"]
    p_Y_train = p_experiment / config["p_Y_train"]
    p_R_valid = p_experiment / config["p_R_valid"]
    p_Y_valid = p_experiment / config["p_Y_valid"]

    # TRAIN
    ds_train = ClassificationDataset(
        p_data = config.p_data,
        transform=transform_linprob,
        target_transform=None,
        split="train"
    )
    dl_train = DataLoader(
        ds_train,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=False,
        drop_last=True
    )
    # LINPROB
    ds_valid = ClassificationDataset(
        p_data = config.p_data,
        transform=transform_linprob,
        target_transform=None,
        split="valid"
    )
    dl_valid = DataLoader(
        ds_valid,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers = config.num_workers,
        pin_memory=False
    )

    model.eval()
    R_train, Y_train = utils.get_representations(model.backbone, dl_train, device)
    R_valid, Y_valid = utils.get_representations(model.backbone, dl_valid, device)
    #
    np.save(p_R_train, R_train)
    np.save(p_Y_train, Y_train)
    np.save(p_R_valid, R_valid)
    np.save(p_Y_valid, Y_valid)

    # EVAL with all Features
    ds_eval_train = ClassificationDataset(
        p_data = config.p_data,
        transform=transform_linprob,
        target_transform=None,
        split="train"
    )
    dl_eval_train = DataLoader(
        ds_eval_train,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers = config.num_workers,
        pin_memory=False
    )
    ds_eval_valid = ClassificationDataset(
        p_data = config.p_data,
        transform=transform_linprob,
        target_transform=None,
        split="valid"
    )
    dl_eval_valid = DataLoader(
        ds_eval_valid,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers = config.num_workers,
        pin_memory=False
    )

    def get_representation(x):
        return model.backbone(x)

    R_train, Y_train = utils.get_representations(get_representation, dl_eval_train, device, imgs=False)
    print(R_train.shape, Y_train.shape)

    R_valid, Y_valid, X_valid = utils.get_representations(get_representation, dl_eval_valid, device, imgs=True, inverse_norm_transform=inverse_norm_transform)
    print(R_valid.shape, Y_valid.shape, X_valid.shape)



    results = []
    for target_idx in range(Y_valid.shape[1]):
        target = ds_config["classes"][target_idx]
        if len(set(Y_train[:, target_idx])) == 1:
            print("{:>15}: acc = NA".format(target))
            results.append(np.inf)
            continue
        clf = LogisticRegression(random_state=0).fit(R_train, Y_train[:, target_idx])
        score = clf.score(R_valid, Y_valid[:, target_idx])
        target = ds_config["classes"][target_idx]
        print("{:>15}: acc = {:.2f}".format(target, score))
        results.append(score)

    fig, ax = plt.subplots(1, 1)
    ax.bar(range(len(results)), results, width=1)
    ax.set_ylim([0, 1])
    ax.set_xticks(np.arange(len(ds_config["classes"])))
    ax.set_xticklabels(ds_config["classes"])
    plt.title("Prediction Accurace LR on valid")
    plt.savefig(p_experiment / "score_lr.png")
    plt.close()

    n_plot = 100
    idcs = np.random.choice(R_valid.shape[0], size=n_plot, replace=False)
    #
    R_plot = R_valid[idcs]
    Y_plot = Y_valid[idcs]
    #
    dim_featuers = R_plot.shape[1]
    num_targets = Y_plot.shape[1]
    scale = 4
    figsize = (num_targets * scale, dim_featuers)
    fig, axes = plt.subplots(1, num_targets, figsize=figsize)
    for col_idx in range(num_targets):
        ax = axes[col_idx]
        ax.set_title(ds_config["classes"][col_idx])
        for row_idx in range(dim_featuers):
            # reps
            r = R_plot[:, row_idx]
            r = (r - r.min()) / (r - r.min()).max()
            # targets
            y = Y_plot[:,col_idx]
            xx = np.ones(len(r)) * row_idx
            #
            ax.scatter(r, xx, c=y, cmap="turbo")
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            #ax.set_ylim([0.95, 1.05])
    plt.savefig(p_experiment / "class_distribution.png")
    plt.tight_layout()
    plt.close()

    R = R_valid
    plt.bar(range(R.shape[1]), R.mean(axis=0), width=1)
    plt.title("Feature Mean")
    plt.savefig(p_experiment / "feature_dist_valid.png")
    plt.close()

    plt.bar(range(R.shape[0]), R.mean(axis=1), width=1)
    plt.title("Sample Mean")
    plt.savefig(p_experiment / "sample_dist_valid.png")
    plt.close()

    R = R_valid
    X = X_valid
    Y = Y_valid
    #
    n_imgs = 50
    topic_idcs = []
    for dim_idx in range(R.shape[1]):
        r = R[:, dim_idx]
        idcs = np.argsort(r)[-n_imgs:]
        topic_idcs.append(idcs)
    topic_idcs = np.array(topic_idcs)


    h, w = np.array(topic_idcs.shape) * 64
    img = np.zeros((h, w, 3))
    print(img.shape)
    n_rows, n_cols = topic_idcs.shape
    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            img_idx = topic_idcs[row_idx][col_idx]
            img[row_idx * 64: row_idx * 64 + 64, col_idx * 64:col_idx * 64 + 64,:] = X[img_idx]
    Image.fromarray(np.uint8(img * 255)).save(p_experiment / "feature_dims_highest.png")



if __name__ == "__main__":

    all_stages = []
    for cnt_overlap in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]:
        for ratio_stl_geo in [0.2, 0.4, 0.6, 0.8]:
            all_stages.append((cnt_overlap, ratio_stl_geo))

    all_stages = all_stages[::-1]
    step = 1
    device = 1
    for cnt_overlap, ratio_stl_geo in all_stages:
        print("#" * 100)
        print("[{:>3d} /{:>3d}]: cnt_overlap: {}, ratio_stl_geo: {}".format(step, len(all_stages), ratio_stl_geo, ratio_stl_geo))
        print("#" * 100)
        config = {
            'device': 'cuda',
            'cuda_visible_devices': "{}".format(device),
            'p_data': '/mnt/data/csprites/single_csprites_64x64_n7_c32_a32_p30_s3_bg_inf_random_function_70000',
            'target_variable': 'shape',
            'batch_size': 512,
            'num_workers': 20,
            'num_epochs': 60,
            'freqs': {
                'ckpt': 50,         # epochs
                'linprob': 5,       # epochs
            },
            'num_vis': 64,
            'backbone': "FCN16i223o64",
            'backbone_args': {
                'ch_last': 64,
                'dim_in': 3,
            },
            'dim_out': 64,
            'optimizer': 'adam',
            'optimizer_args': {
                'lr': 0.001,
                'weight_decay': 1e-6
            },
            'projector': [256,
                          256,
                          256],
            'cnt_overlap': cnt_overlap,
            'ratio_stl_geo': ratio_stl_geo,
            'p_ckpts': "ckpts",
            'p_model': "model_{}.ckpt",
            'p_stats': "stats.pkl",
            'p_config': 'config.pkl',
            'p_R_train': 'R_train.npy',
            'p_R_valid': 'R_valid.npy',
            'p_Y_valid': 'Y_valid.npy',
            'p_Y_train': 'Y_train.npy',
        }
        p_base = Path("/mnt/experiments/csprites") / Path(config["p_data"]).name / "BTL3"
        #
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M-%S')
        #
        config["p_experiment"] = str(p_base / "BT_[{}_d{}]_[L3_stl_geo_{}_ol_{}]_{}".format(
            config["backbone"],
            config["backbone_args"]["ch_last"],
            ratio_stl_geo,
            cnt_overlap,
            st
            )
                                    )
        config = DottedDict(config)
        main(config)
        step += 1
