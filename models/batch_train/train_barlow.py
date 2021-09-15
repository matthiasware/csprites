#!/usr/bin/env python
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
from PIL import Image
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
import eval_utils


def main(config):
    pprint.pprint(config)
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

    transform_train = transforms.Compose([
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
        p_data = config.p_data,
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
        p_data = config.p_data,
        transform=transform_linprob,
        target_transform=target_transform,
        split="valid"
    )
    dl_linprob = DataLoader(
        ds_linprob,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers = config.num_workers,
        pin_memory=False
    )

    model = BarlowTwins(get_backbone(config.backbone, **config.backbone_args),
                        config.projector,
                        config.lambd,
                        config.scale_factor)
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

    global_step = 0
    for epoch_idx in range(1, config.num_epochs + 1, 1):
        ################
        # TRAIN
        ################
        model.train()
        epoch_step = 0
        epoch_loss = 0
       
        desc = "Epoch [{:3}/{:3}] {}:".format(epoch_idx, config.num_epochs, 'train')
        pbar = tqdm(dl_train, bar_format= desc + '{bar:10}{r_bar}{bar:-10b}')
        #
        for (x1, x2), _ in pbar:
            x1 = x1.to(device)
            x2 = x2.to(device)
            for param in model.parameters():
                param.grad = None
            loss, on_diag, off_diag = model.forward(x1, x2, return_all=True)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_step += 1
            global_step += 1
            #
            pbar.set_postfix({'loss': loss.item(), "on_diag": on_diag.item(), "off_diag": off_diag.item()})

        stats.train.loss.append(epoch_loss / epoch_step)
        stats.train.epoch.append(epoch_idx)

        ################
        # Linprob
        ################
        if epoch_idx % config.freqs.linprob == 0 or epoch_idx == config.num_epochs:
            model.eval()
            linacc, knnacc = utils.linprob_model(model.backbone, dl_linprob, device)
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


    dl_train, dl_valid = utils.get_raw_csprites_dataloader(
        p_data=config.p_data,
        img_size = ds_config["img_size"],
        batch_size = config.batch_size,
        norm_transform=norm_transform,
        num_workers = config["num_workers"]
    )
    p_R_train = p_experiment / config["p_R_train"]
    p_Y_train = p_experiment / config["p_Y_train"]
    p_R_valid = p_experiment / config["p_R_valid"]
    p_Y_valid = p_experiment / config["p_Y_valid"]
    #
    model.eval()
    R_train, Y_train = utils.get_representations(model.backbone, dl_train, device, imgs=False)
    R_valid, Y_valid, X_valid = utils.get_representations(model.backbone, dl_valid, device, imgs=True, inverse_norm_transform=inverse_norm_transform)
    #
    np.save(p_R_train, R_train)
    np.save(p_Y_train, Y_train)
    np.save(p_R_valid, R_valid)
    np.save(p_Y_valid, Y_valid)

    #
    print("TRAIN (R, Y)", R_train.shape, Y_train.shape)
    print("VALID (R, Y)", R_valid.shape, Y_valid.shape)

    eval_utils.eval_representations(
        R_train=R_train,
        R_valid=R_valid,
        Y_train=Y_train,
        Y_valid=Y_valid,
        X_valid=X_valid,
        p_experiment=p_experiment,
        class_names = ds_config["classes"],
        show=False
    )


if __name__ == "__main__":
    device = 0
    bb_config = []
    all_backbones = ["FCN32i223o128", "FCN16i223o64", "FCN8i223o32"]
    all_chlasts = [256, 128, 64, 32]
    for backbone in all_backbones:
        for ch_last in all_chlasts:
            bb_config.append((backbone, ch_last))

    step = 1
    for backbone, ch_last in bb_config:
        print("#" * 100)
        print("[{:>3d} /{:>3d}]: {}: {}".format(step, len(bb_config), backbone, ch_last))
        print("#" * 100)
        config = {
            'device': 'cuda',
            'cuda_visible_devices': "{}".format(device),
            'p_data': "/mnt/data/csprites/single_csprites_64x64_n7_c24_a32_p13_s3_bg_inf_random_function_77000",
            'target_variable': 'shape',
            'batch_size': 512,
            'num_workers': 20,
            'num_epochs': 100,
            'freqs': {
                'ckpt': 200,        # epochs
                'linprob': 10,       # epochs
            },
            'num_vis': 64,
            'backbone': backbone,
            'backbone_args': {
                'ch_last': ch_last,
                'dim_in': 3,
            },
            'optimizer': 'adam',
            'optimizer_args': {
                'lr': 0.001,
                'weight_decay': 1e-6
            },
            'projector': [4 * ch_last, 4 * ch_last, 4 * ch_last],
            'scale_factor': 1,
            'p_ckpts': "ckpts",
            'p_model': "model_{}.ckpt",
            'p_stats': "stats.pkl",
            'p_config': 'config.pkl',
            'p_R_train': 'R_train.npy',
            'p_R_valid': 'R_valid.npy',
            'p_Y_valid': 'Y_valid.npy',
            'p_Y_train': 'Y_train.npy',
            'p_R_train_barlow': 'R_train.npy',
            'p_R_valid_barlow': 'R_valid.npy',
            'p_Y_valid_barlow': 'Y_valid.npy',
            'p_Y_train_barlow': 'Y_train.npy',
        }
        p_base = Path("/mnt/experiments/csprites") / Path(config["p_data"]).name / "BarlowTwins"
        #
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M-%S')
        #
        config["p_experiment"] = str(p_base / "BTwins_[{}_d_{}]_{}".format(config["backbone"],
                                                                           config["backbone_args"]["ch_last"],
                                                                           st))
        config['lambd'] = calc_lambda(config["projector"][-1])
        config = DottedDict(config)
        step += 1

        try:
            main(config)
        except Exception as e:
            print("ERROR but continue")
