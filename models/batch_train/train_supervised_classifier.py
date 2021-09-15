import sys
sys.path.append("..")
sys.path.append("../..")

import os
from pathlib import Path
import pickle
import timeit

import warnings
warnings.filterwarnings('ignore')
#
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, models
from dotted_dict import DottedDict
from tqdm import tqdm
import pprint
#
from csprites.datasets import ClassificationDataset
import utils
from backbone import get_backbone
from optimizer import get_optimizer
import plot_utils
import eval_utils


def main(config):
    print("#" * 100)
    print(config.target_variable)
    print("#" * 100)
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
    #
    norm_transform = utils.normalize_transform(ds_config["means"],
                                   ds_config["stds"])
    #
    target_transform = lambda x: x[target_idx]
    transform = transform = transforms.Compose(
        [transforms.ToTensor(),
         norm_transform,
        ])
    inverse_norm_transform = utils.inverse_normalize_transform(
        ds_config["means"],
        ds_config["stds"]
    )

    # TRAIN
    ds_train = ClassificationDataset(
        p_data = config.p_data,
        transform=transform,
        target_transform=target_transform,
        split="train"
    )
    dl_train = DataLoader(
        ds_train,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=False
    )
    # VALID
    ds_valid = ClassificationDataset(
        p_data = config.p_data,
        transform=transform,
        target_transform=target_transform,
        split="valid"
    )
    dl_valid = DataLoader(
        ds_valid,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers = config.num_workers,
        pin_memory=False
    )

    model = get_backbone(config.backbone, **config.backbone_args)
    model.fc = torch.nn.Linear(in_features=model.dim_out, out_features=n_classes)
    print("#param [m]: {:.3f}".format(utils.count_parameters(model) * 1e-6))
    #
    if torch.cuda.device_count() > 1 and device != "cpu":
        print("Using {} gpus!".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    print(model)

    optimizer = get_optimizer(config.optimizer, model.parameters(), config.optimizer_args)
    criterion = nn.CrossEntropyLoss()

    stats = {
        'train': {
            'loss': [],
            'acc': [],
            'epoch': [],
        },
        'valid': {
            'loss': [],
            'acc': [],
            'epoch': [],
        }
    }
    stats = DottedDict(stats)

    p_experiment = Path(config.p_experiment)
    p_experiment.mkdir(exist_ok=True, parents=True)
    p_ckpts = p_experiment / config.p_ckpts
    p_ckpts.mkdir(exist_ok=True)

    print_tmp = "    Epoch [{:3}/{:3}] - {}: loss: {:.3f} acc: {:.3f}"
    desc_tmp = "Epoch [{:3}/{:3}] {}:"
    #
    for epoch_idx in range(1, config.num_epochs + 1, 1):
        ################
        # TRAIN
        ################
        model.train()
        epoch_step = 0
        epoch_loss = 0
        epoch_total = 0
        epoch_correct = 0
        #
        desc = desc_tmp.format(epoch_idx, config.num_epochs, 'train')
        pbar = tqdm(dl_train, bar_format= desc + '{bar:10}{r_bar}{bar:-10b}')
        #
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)
            for param in model.parameters():
                param.grad = None
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            #
            _, y_pred = torch.max(out, 1)
            total = y.size(0)
            correct = (y_pred == y).sum().item()
            #
            epoch_loss += loss.item()
            epoch_total += total
            epoch_correct += correct
            epoch_step += 1
            #
            pbar.set_postfix({'loss': loss.item(), 'acc': correct / total})

        stats.train.loss.append(epoch_loss / epoch_step)
        stats.train.acc.append(epoch_correct / epoch_total)
        stats.train.epoch.append(epoch_idx)
        
        ################
        # EVAL
        ################
        if epoch_idx % config.freqs.eval == 0 or epoch_idx == config.num_epochs:
            model.eval()
            epoch_step = 0
            epoch_loss = 0
            epoch_total = 0
            epoch_correct = 0
            #
            desc = desc_tmp.format(epoch_idx, config.num_epochs, 'valid')
            pbar = tqdm(dl_valid, bar_format= desc + '{bar:10}{r_bar}{bar:-10b}')
            #
            for x, y in pbar:
                x = x.to(device)
                y = y.to(device)
                with torch.no_grad():
                    out = model(x)
                    loss = criterion(out, y)
                #
                    _, y_pred = torch.max(out, 1)
                total = y.size(0)
                correct = (y_pred == y).sum().item()
                #
                epoch_loss += loss.item()
                epoch_total += total
                epoch_correct += correct
                epoch_step += 1
                #
                pbar.set_postfix({'loss': loss.item(), 'acc': correct / total})
            #
            stats.valid.loss.append(epoch_loss / epoch_step)
            stats.valid.acc.append(epoch_correct / epoch_total)
            stats.valid.epoch.append(epoch_idx)
        if epoch_idx % config.freqs.ckpt == 0 or epoch_idx == config.num_epochs:
            print("save model!")
            if torch.cuda.device_count() > 1 and device != "cpu":
                torch.save(model.module.state_dict(), p_ckpts / config.p_model.format(epoch_idx))
            else:
                torch.save(model.state_dict(), p_ckpts / config.p_model.format(epoch_idx))

    # plot losses
    plt.plot(stats.train.epoch, stats.train.loss, label="train")
    plt.plot(stats.valid.epoch, stats.valid.loss, label="valid")
    plt.yscale('log')
    plt.legend()
    plt.savefig(p_experiment / "loss.png")
    plt.close()

    # plot accs
    plt.plot(stats.train.epoch, stats.train.acc, label="train")
    plt.plot(stats.valid.epoch, stats.valid.acc, label="valid")
    plt.yticks([0.5,0.6,0.7,0.8,0.85,0.9,0.95,1])
    plt.legend()
    plt.savefig(p_experiment / "acc.png")
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

    model.fc = nn.Identity()
    model.eval()
    R_train, Y_train = utils.get_representations(model, dl_train, device, imgs=False)
    R_valid, Y_valid, X_valid = utils.get_representations(model, dl_valid, device, imgs=True, inverse_norm_transform=inverse_norm_transform)
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
    bb_config = []
    for backbone in ["FCN16i223o64"]:
        for ch_last in [64]:
            for target in ["shape", "scale", "color", "angle", "py", "px"]:
                bb_config.append((backbone, ch_last, target))

                config = {
                    'device': 'cuda',
                    'cuda_visible_devices': '0',
                    'p_data': "/mnt/data/csprites/single_csprites_64x64_n7_c24_a32_p13_s3_bg_inf_random_function_77000",
                    'target_variable': target,
                    'batch_size': 512,
                    'num_workers': 6,
                    'num_epochs': 30,
                    'freqs': {
                        'ckpt': 50,         # epochs
                        'eval': 5,          # epochs
                    },
                    'num_vis': 64,
                    'backbone': backbone,
                    'dim_out': ch_last,
                    'backbone_args':
                    {
                        'ch_last': ch_last,
                        'dim_in': 3,
                    },
                    'optimizer': 'adam',
                    'optimizer_args':
                    {
                        'lr': 0.001,
                        # weight_decay': 1e-5
                    },
                    'p_ckpts': "ckpts",
                    'p_model': "model_{}.ckpt",
                    'p_stats': "stats.pkl",
                    'p_config': 'config.pkl',
                    'p_R_train': 'R_train.npy',
                    'p_R_valid': 'R_valid.npy',
                    'p_Y_valid': 'Y_valid.npy',
                    'p_Y_train': 'Y_train.npy'
                }
                p_base = Path("/mnt/experiments/csprites") / \
                    Path(config["p_data"]).name
                config["p_experiment"] = str(p_base / "SUP_[{}_d{}]_target_[{}]".format(
                    config["backbone"],
                    config["dim_out"],
                    config["target_variable"]))
                config = DottedDict(config)
                pprint.pprint(config)
                main(config)
