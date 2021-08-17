import sys
sys.path.append("..")

import os
from pathlib import Path
import pickle
import timeit
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
    target_transform = lambda x: x[target_idx]
    transform = transforms.Compose(
        [transforms.ToTensor(),
         utils.normalize_transform(ds_config["means"],
                                   ds_config["stds"])
        ])

    # DS
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

    model = get_backbone(config.backbone, pretrained=False)
    model.fc = torch.nn.Linear(in_features=model.dim_out, out_features=n_classes)
    print("#param [m]: {:.3f}".format(utils.count_parameters(model) * 1e-6))
    #
    if torch.cuda.device_count() > 1 and device != "cpu":
        print("Using {} gpus!".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    #
    optimizer = get_optimizer(config.optimizer, model, config.optimizer_args)
    criterion = nn.CrossEntropyLoss()

    # TRAIN
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

    # plot accs
    plt.plot(stats.train.epoch, stats.train.acc, label="train")
    plt.plot(stats.valid.epoch, stats.valid.acc, label="valid")
    plt.yscale('log')
    plt.legend()
    plt.savefig(p_experiment / "acc.png")

    with open(p_experiment / config.p_config, "wb") as file:
        pickle.dump(config, file)
    with open(p_experiment / config.p_stats, "wb") as file:
        pickle.dump(stats, file)

    # TRAIN
    ds_train = ClassificationDataset(
        p_data = config.p_data,
        transform=transform,
        target_transform=None,
        split="train"
    )
    dl_train = DataLoader(
        ds_train,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=False
    )
    # VALID
    ds_valid = ClassificationDataset(
        p_data = config.p_data,
        transform=transform,
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

    model = get_backbone(config.backbone, pretrained=False)
    ckpt = torch.load(str(p_ckpts / config.p_model.format(config.num_epochs)))
    r = model.load_state_dict(ckpt, strict=False)
    assert len(r.missing_keys) == 0
    assert len(r.unexpected_keys) == 2
    #
    model = model.to(device)
    model.eval()

    p_R_train = p_experiment / config["p_R_train"]
    p_Y_train = p_experiment / config["p_Y_train"]
    p_R_valid = p_experiment / config["p_R_valid"]
    p_Y_valid = p_experiment / config["p_Y_valid"]

    R_train = []
    R_valid = []
    Y_train = []
    Y_valid = []
    #
    for x, y in tqdm(dl_train):
        x = x.to(device)
        with torch.no_grad():
            r = model(x).detach().cpu().numpy()
        R_train.append(r)
        Y_train.append(y.numpy())
    #
    for x, y in tqdm(dl_valid):
        x = x.to(device)
        with torch.no_grad():
            r = model(x).detach().cpu().numpy()
        R_valid.append(r)
        Y_valid.append(y.numpy())

    R_train = np.concatenate(R_train)
    R_valid = np.concatenate(R_valid)
    Y_train = np.concatenate(Y_train)
    Y_valid = np.concatenate(Y_valid)

    np.save(p_R_train, R_train)
    np.save(p_Y_train, Y_train)
    np.save(p_R_valid, R_valid)
    np.save(p_Y_valid, Y_valid)


if __name__ == "__main__":
    for target in ["shape", "scale", "color", "angle", "py", "px"]:
        config = {
            'device': 'cuda',
            'cuda_visible_devices': '0,1',
            'p_data': "/mnt/data/csprites/single_csprites_64x64_n7_c16_a16_p8_s4_bg_inf_random_function_458752",
            'target_variable': target,
            'batch_size': 1024,
            'num_workers': 6,
            'num_epochs': 10,
            'freqs': {
                'ckpt': 10,         # epochs
                'eval': 1,          # epochs
            },
            'num_vis': 64,
            'backbone': "ResNet-18",
            'optimizer': 'adam',
            'optimizer_args': {
                'lr': 0.0005,
                'weight_decay': 1e-6
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
        config["p_experiment"] = str(p_base / "bb_[{}]_target_[{}]".format(config["backbone"],
                                                                           config["target_variable"]))
        config = DottedDict(config)
        main(config)
