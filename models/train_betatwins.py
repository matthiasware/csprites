import sys
sys.path.append("..")

import numpy as np
import time
import torch
import torch.nn as nn
from torch.distributions import Beta
from torch.distributions.dirichlet import Dirichlet
from tqdm import tqdm
import torch.nn.functional as F
#
from utils import *
# Python
from pathlib import Path
import os
import warnings
import math
import datetime
import time
warnings.filterwarnings('ignore')

# Extern
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.resnet import resnet18
from dotted_dict import DottedDict
import pickle
from tqdm import tqdm
import pprint
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# Local
from BTwins.barlow import *
from BTwins.transform_utils import *
from csprites.datasets import ClassificationDataset
import utils
from backbone import get_backbone
from optimizer import get_optimizer
#
from utils import linprob_model, get_representations
from Beta.utils import *
from Beta.models import BetaTwins


def main(depth_bp, width_bp, weight_dkl, alpha, beta, device):
    config = {
        'device': 'cuda',
        'cuda_visible_devices': str(device),
        'p_data': '/mnt/data/csprites/single_csprites_64x64_n7_c128_a32_p10_s3_bg_inf_random_function_100000',
        'target_variable': 'shape',
        'batch_size': 1024,
        'num_workers': 6,
        'num_epochs': 50,
        'freqs': {
            'ckpt': 50,         # epochs
            'linprob': 5,       # epochs
        },
        'num_vis': 64,
        'backbone': "ResNet-18",
        'optimizer': 'adam',
        'optimizer_args': {
            'lr': 0.001,
            'weight_decay': 1e-6
        },
        'projector': [1024, 1024, 1024],
        'backbone_projector': [width_bp] * depth_bp,
        'alpha': alpha,
        'beta': beta,
        'w_dkl': weight_dkl,
        'w_sim': 1,
        'p_ckpts': "ckpts",
        'p_model': "model_{}.ckpt",
        'p_stats': "stats.pkl",
        'p_config': 'config.pkl',
        'p_R_train': 'R_train.npy',
        'p_R_valid': 'R_valid.npy',
        'p_Y_valid': 'Y_valid.npy',
        'p_Y_train': 'Y_train.npy',
        'p_R_train_bp': 'R_train_bp.npy',
        'p_R_valid_bp': 'R_valid_bp.npy',
        'p_Y_valid_bp': 'Y_valid_bp.npy',
        'p_Y_train_bp': 'Y_train_bp.npy',

    }
    p_base = Path("/mnt/experiments/csprites") / \
        Path(config["p_data"]).name / "beta"
    #
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M-%S')
    #
    config["p_experiment"] = str(p_base / "Beta_w{}_d{}_alpha_{}_beta_{}_wdkl_{}_[{}]_{}".format(
        config["backbone_projector"][-1],
        len(config["backbone_projector"]),
        config["alpha"],
        config["beta"],
        config["w_dkl"],
        config["backbone"],
        st))
    config = DottedDict(config)
    pprint.pprint(config)

    # TORCH SETTINGS
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices
    device = torch.device(config.device)

    p_ds_config = Path(config.p_data) / "config.pkl"

    with open(p_ds_config, "rb") as file:
        ds_config = pickle.load(file)

    target_variable = config.target_variable
    target_idx = [idx for idx, target in enumerate(
        ds_config["classes"]) if target == target_variable][0]
    n_classes = ds_config["n_classes"][target_variable]

    norm_transform = utils.normalize_transform(
        ds_config["means"],
        ds_config["stds"])
    inverse_norm_transform = utils.inverse_normalize_transform(
        ds_config["means"],
        ds_config["stds"]
    )
    def target_transform(x): return x[target_idx]

    transform_train = transforms.Compose([
        # transforms.Resize(ds_config["img_size"]),
        transforms.RandomResizedCrop(ds_config["img_size"],
                                     scale=(0.6, 1.0),
                                     ratio=(1, 1),
                                     interpolation=Image.BICUBIC),
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
        norm_transform
    ])
    transform_train_prime = transforms.Compose([
        # transforms.Resize(ds_config["img_size"]),
        transforms.RandomResizedCrop(ds_config["img_size"],
                                     scale=(0.6, 1.0),
                                     ratio=(1, 1),
                                     interpolation=Image.BICUBIC),
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
        norm_transform
    ])

    transform_linprob = transforms.Compose([
        transforms.Resize(ds_config["img_size"]),
        transforms.ToTensor(),
        norm_transform
    ])

    # TRAIN
    ds_train = ClassificationDataset(
        p_data=config.p_data,
        transform=Transform(transform_train, transform_train_prime),
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
    ds_linprob = ClassificationDataset(
        p_data=config.p_data,
        transform=transform_linprob,
        target_transform=target_transform,
        split="valid"
    )
    dl_linprob = DataLoader(
        ds_linprob,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=False
    )

    model = BetaTwins(
        get_backbone(config.backbone, pretrained=False,
                     zero_init_residual=True),
        config.projector,
        config.backbone_projector
    )
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
    print(model.backbone_projector)
    print(model.projector)
    #
    optimizer = get_optimizer(
        config.optimizer, model.parameters(), config.optimizer_args)

    stats = {
        'train': {
            'loss': [],
            'loss_sim': [],
            'loss_dkl': [],
            'epoch': [],
        },
        'linprob': {
            'bb': {
                'linacc': [],
                'knnacc': [],
                'epoch': [],
            },
            'bp': {
                'linacc': [],
                'knnacc': [],
                'epoch': [],
            }
        }
    }
    stats = DottedDict(stats)
    #
    p_experiment = Path(config.p_experiment)
    p_experiment.mkdir(exist_ok=True, parents=True)
    p_ckpts = p_experiment / config.p_ckpts
    p_ckpts.mkdir(exist_ok=True)
    print(config.p_experiment)

    alpha_true = torch.Tensor([config["alpha"]]).to(device)
    beta_true = torch.Tensor([config["beta"]]).to(device)
    #
    print(alpha_true, beta_true)
    #
    lmbda = calc_lambda(config.projector[-1])
    print("LMBDA: {:.5f}".format(lmbda))
    #
    print("W_SIM: {}\nW_DKL: {}".format(config.w_sim, config.w_dkl))
    print("D_BP:  {}".format(len(config.backbone_projector)))
    print("W_BP:  {}".format(config.backbone_projector[-1]))

    global_step = 0
    for epoch_idx in range(1, config.num_epochs + 1, 1):
        ################
        # TRAIN
        ################
        model.train()
        epoch_step = 0
        epoch_loss = 0
        epoch_loss_sim = 0
        epoch_loss_dkl = 0

        desc = "Epoch [{:3}/{:3}]:".format(epoch_idx, config.num_epochs)
        pbar = tqdm(dl_train, bar_format=desc + '{bar:10}{r_bar}{bar:-10b}')
        #
        for (x1, x2), _ in pbar:
            x1 = x1.to(device)
            x2 = x2.to(device)
            for param in model.parameters():
                param.grad = None
            z1 = model.representations(x1)
            z2 = model.representations(x2)
            # #############
            # DKL Loss
            # #############
            if config.w_dkl == 0:
                with torch.no_grad():
                    a1, b1 = beta_params(z1)
                    a2, b2 = beta_params(z2)
                    #
                    dkl_1 = kl_beta_beta(
                        (a1, b1), (alpha_true, beta_true)).mean()
                    dkl_2 = kl_beta_beta(
                        (a2, b2), (alpha_true, beta_true)).mean()
                    #
                    loss_dkl = (dkl_1 + dkl_2) * config.w_dkl
            else:
                a1, b1 = beta_params(z1)
                a2, b2 = beta_params(z2)
                #
                dkl_1 = kl_beta_beta((a1, b1), (alpha_true, beta_true)).mean()
                dkl_2 = kl_beta_beta((a2, b2), (alpha_true, beta_true)).mean()
                #
                loss_dkl = (dkl_1 + dkl_2) * config.w_dkl
            #
            # #############
            # BARLOW Loss
            # #############
            z1 = model.projector(z1)
            z2 = model.projector(z2)

            # empirical cross-correlation matrix
            c = model.bn(z1).T @ model.bn(z2)
            c.div_(z1.shape[0])
            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
            off_diag = off_diagonal(c).pow_(2).sum()
            #
            loss_sim = (on_diag + lmbda * off_diag) * config.w_sim
            #
            loss = loss_sim + loss_dkl
            #
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_loss_sim += loss_sim.item()
            epoch_loss_dkl += loss_dkl.item()
            epoch_step += 1
            global_step += 1
            #
            pbar.set_postfix(
                {'loss': loss.item(),
                 'sim': loss_sim.item(),
                 'dkl': loss_dkl.item(),
                 'kl1': dkl_1.item(),
                 'kl2': dkl_2.item(),
                 }
            )

        stats.train.loss.append(epoch_loss / epoch_step)
        stats.train.loss_sim.append(epoch_loss_sim / epoch_step)
        stats.train.loss_dkl.append(epoch_loss_dkl / epoch_step)
        stats.train.epoch.append(epoch_idx)

        ################
        # Linprob
        ################
        if epoch_idx % config.freqs.linprob == 0 or epoch_idx == config.num_epochs:
            model.eval()
            linacc, knnacc = linprob_model(
                model.representations, dl_linprob, device)
            stats.linprob.bp.epoch.append(epoch_idx)
            stats.linprob.bp.knnacc.append(knnacc)
            stats.linprob.bp.linacc.append(linacc)
            print(
                "    Linprob REP @LR: {:.2f} @KNN: {:.2f}".format(linacc, knnacc))
            #
            linacc, knnacc = linprob_model(model.backbone, dl_linprob, device)
            stats.linprob.bb.epoch.append(epoch_idx)
            stats.linprob.bb.knnacc.append(knnacc)
            stats.linprob.bb.linacc.append(linacc)
            print(
                "    Linprob BB  @LR: {:.2f} @KNN: {:.2f}".format(linacc, knnacc))

            model.train()
        # Checkpoint
        if epoch_idx % config.freqs.ckpt == 0 or epoch_idx == config.num_epochs:
            print("save model!")
            if torch.cuda.device_count() > 1 and device != "cpu":
                torch.save(model.module.state_dict(), p_ckpts /
                           config.p_model.format(epoch_idx))
            else:
                torch.save(model.state_dict(), p_ckpts /
                           config.p_model.format(epoch_idx))

    alpha, beta = config["alpha"], config["beta"]
    dist = Beta(alpha, beta)
    plot_beta_pdf(dist, title="GW", p_file=p_experiment / "beta_dist.png")

    # plot losses
    plt.plot(stats.train.epoch, stats.train.loss, label="train")
    plt.title("Loss")
    plt.legend()
    plt.savefig(p_experiment / "loss.png")
    plt.close()

    # plot losses
    plt.plot(stats.train.epoch, stats.train.loss_sim, label="train")
    plt.title("SIM")
    plt.legend()
    plt.savefig(p_experiment / "barlow_loss.png")
    plt.close()

    # plot losses
    plt.plot(stats.train.epoch, stats.train.loss_dkl, label="train")
    plt.title("DKL")
    plt.legend()
    plt.savefig(p_experiment / "beta_loss.png")
    plt.close()

    # plot knn acc
    plt.plot(stats.linprob.bp.epoch, stats.linprob.bp.knnacc, label="Beta")
    plt.plot(stats.linprob.bb.epoch, stats.linprob.bb.knnacc, label="Backbone")
    plt.ylim([0, 1])
    plt.yticks([.1, .2, 0.3, .4, .5, .6, .7, .8, .9, .95, 1])
    plt.legend()
    plt.savefig(p_experiment / "knnaccs.png")
    plt.close()

    # plot lr acc
    plt.plot(stats.linprob.bp.epoch, stats.linprob.bp.linacc, label="Beta")
    plt.plot(stats.linprob.bb.epoch, stats.linprob.bb.linacc, label="Backbone")
    plt.ylim([0, 1])
    plt.yticks([.1, .2, 0.3, .4, .5, .6, .7, .8, .9, .95, 1])
    plt.legend()
    plt.savefig(p_experiment / "linaccs.png")
    plt.close()

    with open(p_experiment / config.p_config, "wb") as file:
        pickle.dump(config, file)
    with open(p_experiment / config.p_stats, "wb") as file:
        pickle.dump(stats, file)

    model.eval()
    linacc, knnacc = linprob_model(model.backbone, dl_linprob, device)
    print("Linprob BACKBONE @LR: {:.2f} @KNN: {:.2f}".format(linacc, knnacc))
    #
    linacc, knnacc = linprob_model(model.representations, dl_linprob, device)
    print("Linprob BETAPROJ @LR: {:.2f} @KNN: {:.2f}".format(linacc, knnacc))

    R_bp, Y_bp = get_representations(model.representations, dl_linprob, device)
    Z = torch.Tensor(R_bp)
    alphas_bp, betas_bp = beta_params(Z)

    xx = range(alphas_bp.shape[0])
    #
    plt.figure(figsize=(12, 2))
    plt.bar(xx, alphas_bp)
    plt.title("alphas")
    plt.savefig(p_experiment / "alphas.png")
    plt.close()
    #
    plt.figure(figsize=(12, 2))
    plt.bar(xx, betas_bp)
    plt.title("betas")
    plt.savefig(p_experiment / "betas.png")
    plt.close()

    p_R_train = p_experiment / config["p_R_train"]
    p_Y_train = p_experiment / config["p_Y_train"]
    p_R_valid = p_experiment / config["p_R_valid"]
    p_Y_valid = p_experiment / config["p_Y_valid"]
    #
    p_R_train_bp = p_experiment / config["p_R_train_bp"]
    p_Y_train_bp = p_experiment / config["p_Y_train_bp"]
    p_R_valid_bp = p_experiment / config["p_R_valid_bp"]
    p_Y_valid_bp = p_experiment / config["p_Y_valid_bp"]
    #
    # TRAIN
    ds_train = ClassificationDataset(
        p_data=config.p_data,
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
        p_data=config.p_data,
        transform=transform_linprob,
        target_transform=None,
        split="valid"
    )
    dl_valid = DataLoader(
        ds_valid,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=False
    )

    R_train, Y_train = get_representations(model.backbone, dl_train, device)
    R_valid, Y_valid = get_representations(model.backbone, dl_valid, device)
    #
    R_train_bp, Y_train_bp = get_representations(
        model.representations, dl_valid, device)
    R_valid_bp, Y_valid_bp = get_representations(
        model.representations, dl_valid, device)
    #
    np.save(p_R_train, R_train)
    np.save(p_Y_train, Y_train)
    np.save(p_R_valid, R_valid)
    np.save(p_Y_valid, Y_valid)
    #
    np.save(p_R_train_bp, R_train_bp)
    np.save(p_Y_train_bp, Y_train_bp)
    np.save(p_R_valid_bp, R_valid_bp)
    np.save(p_Y_valid_bp, Y_valid_bp)


if __name__ == "__main__":
    for width_bp in [8, 16, 32, 64, 128, 256, 512, 1024]:
        for depth_bp in [1, 2, 3, 4]:
            for width_fac_pf in [1, 2, 4, 8]
                for depth_pf in [1, 2,3]:
                    print("HI")
            #try:
            #    main(depth_bp=depth_bp, width_bp=width_bp,
            #         weight_dkl=10, alpha=0.5, beta=0.5, device=0)
            #except Exception as e:
            #    print("EXCEPTION")

    # w_dkl = 10
    # depth_bp = 1
    # width_bp = 512
    # alpha_betas = [(0.5, 0.5),
    #                (0.1, 0.9),
    #                (0.9, 0.1),
    #                (0.2, 0.8),
    #                (0.8, 0.2)]
    # for alpha, beta in alpha_betas:
    #     try:
    #         main(depth_bp=depth_bp, width_bp=width_bp, weight_dkl=w_dkl, alpha=alpha, beta=beta, device=0)
    #     except Exception as e:
    #         print("EYCEPTION")

    # for alpha, beta in alpha_betas:
    #     try:
    #         main(depth_bp=depth_bp, width_bp=width_bp, weight_dkl=w_dkl, alpha=1/alpha, beta=1/beta, device=0)
    #     except Exception as e:
    #         print("EYCEPTION")

    # experiment over w and d of BP
    #  - w_dkl [32, 64, 128, 256, 512]
    #  - d_dkl [1, 2, 3]
    #  - alpha = beta = 0.5
    #  - w_dkl = 10
    # gpu = 0


width_beta = [0, 1, 2]
depth_beat = [64, 128, 256, 512]
depth_barlow = [0, 1, 2, 3]
depth_factor_beta = [1, 2, 4, 8]
w_dkl = [1, 10, 100]