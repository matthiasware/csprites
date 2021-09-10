import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from dotted_dict import DottedDict


class Net(nn.Module):
    def __init__(self, n_classes, d_in, d_hid=1024, n_hid=0, batch_norm=False):
        super(Net, self).__init__()
        dims = [d_in]
        for _ in range(n_hid):
            dims.append(d_hid)
        dims.append(n_classes)
        #
        layers = []
        for idx in range(1, len(dims) - 1, 1):
            layers.append(nn.Linear(dims[idx - 1], dims[idx]))
            if batch_norm:
                layers.append(nn.BatchNorm1d(dims[idx]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def load_data_tensors(target_idx, p_R_train, p_Y_train, p_R_valid, p_Y_valid):
    R_train = np.load(p_R_train)
    R_valid = np.load(p_R_valid)
    #
    Y_train = np.load(p_Y_train)
    Y_valid = np.load(p_Y_valid)
    #
    d_r = R_train.shape[1]
    #
    Y_train = Y_train[:, target_idx]
    Y_valid = Y_valid[:, target_idx]
    #
    return R_train, Y_train, R_valid, Y_valid


def get_datasets(target_idx, p_R_train, p_R_valid, p_Y_train, p_Y_valid, batch_size):
    #
    R_train = torch.Tensor(np.load(p_R_train))
    R_valid = torch.Tensor(np.load(p_R_valid))
    #
    Y_train = torch.LongTensor(np.load(p_Y_train))
    Y_valid = torch.LongTensor(np.load(p_Y_valid))
    #
    d_r = R_train.shape[1]
    #
    Y_train = Y_train[:, target_idx]
    Y_valid = Y_valid[:, target_idx]
    #
    ds_train = torch.utils.data.TensorDataset(R_train, Y_train)
    ds_valid = torch.utils.data.TensorDataset(R_valid, Y_valid)

    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True)
    dl_valid = DataLoader(
        ds_valid,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True)
    return dl_train, dl_valid, d_r


def train_model(model, num_epochs, optimizer, criterion, dl_train, dl_valid, device):
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
    desc_tmp = "Epoch [{:3}/{:3}] {}:"
    #
    for epoch_idx in range(1, num_epochs + 1, 1):
        ################
        # TRAIN
        ################
        model.train()
        epoch_step = 0
        epoch_loss = 0
        epoch_total = 0
        epoch_correct = 0
        #
        desc = desc_tmp.format(epoch_idx, num_epochs, 'train')
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
        model.eval()
        epoch_step = 0
        epoch_loss = 0
        epoch_total = 0
        epoch_correct = 0
        #
        desc = desc_tmp.format(epoch_idx, num_epochs, 'valid')
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
    return stats