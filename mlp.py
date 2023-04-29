# %%
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader, TensorDataset, SubsetRandomSampler

from utils.epoch_utils import train_epoch, val_epoch, test_epoch


# %% hyper
class HyperClass(object):
    r"""The class to store and handle all the hyper parameters

    Should be sent into main_func
    """

    def __init__(self,
                 seed=20124861,
                 num_epochs=50,
                 batch_size=1024,
                 fold_num=10,
                 information_kept=0.95,
                 if_pca=True,
                 learning_rate=0.002,
                 num_workers=4,
                 num_hidden_layer=2,
                 num_channel=2048,
                 optimizer="adam",
                 exp_path_name="1",
                 base_path="/mnt/emc01/zeyu/mlcw/exp/"
                 ):
        self.seed = seed
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.fold_num = fold_num
        self.information_kept = information_kept
        self.if_pca = if_pca
        self.learning_rate = learning_rate
        self.num_workers = num_workers
        self.num_hidden_layer = num_hidden_layer
        self.num_channel = num_channel
        self.optimizer = optimizer
        self.exp_path_name = exp_path_name
        self.base_path = base_path
        self.exp_path = str(Path(base_path).joinpath(exp_path_name))
        # create dir for experiments
        os.makedirs(self.exp_path, exist_ok=True)

    def save(self):
        data = self.__dict__
        json.dump(data, open(Path(self.exp_path).joinpath("hyper.json"), "w"))


# %%
class MLP(nn.Module):
    r"""Basic MLP model

    """

    def __init__(self, input_channel, num_hidden_layer=2, num_channel=2048, n_class=10):
        super().__init__()

        self.hidden = self._make_layer(input_channel, num_hidden_layer, num_channel)

        self.output_layer = nn.Linear(num_channel, n_class)

    def _make_layer(self, input_channel: int, num_layer: int, num_channel: int) -> nn.Sequential:
        """
            Stack 'num_layer' linear layers with 'input_channel' channels
        """
        layers = []
        layers.append(nn.Linear(input_channel, num_channel))
        layers.append(nn.ReLU(inplace=True))

        if num_layer > 1:
            for _ in range(num_layer - 1):
                layers.append(nn.Linear(num_channel, num_channel))
                layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        # input & hidden
        x = self.hidden(x)

        # output
        x = self.output_layer(x)

        return x


# %%
def main_func(hyper: HyperClass):
    # %% load dataset
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                 download=True, transform=transform)

    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                download=True, transform=transform)

    # %% convert dataset to np.array format
    train_loader_full = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset))
    test_loader_full = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset))

    train_dataset_pair = next(iter(train_loader_full))
    test_dataset_pair = next(iter(test_loader_full))

    X_train = train_dataset_pair[0].numpy().reshape((-1, 1024))
    y_train = train_dataset_pair[1].numpy()

    X_test = next(iter(test_loader_full))[0].numpy().reshape((-1, 1024))
    y_test = test_dataset_pair[1].numpy()

    # %% conduct PCA on the data
    if hyper.if_pca and (hyper.information_kept != 1):
        pca = PCA(n_components=hyper.information_kept)

        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        input_channel = X_train_pca[0].shape[0]

        print('Information kept: ', sum(pca.explained_variance_ratio_) * 100, '%')
        print('Noise variance: ', pca.noise_variance_)
    else:
        input_channel = 32 * 32
        X_train_pca = X_train
        X_test_pca = X_test

    train_dataset = TensorDataset(torch.Tensor(X_train_pca), torch.LongTensor(y_train))
    test_dataset = TensorDataset(torch.Tensor(X_test_pca), torch.LongTensor(y_test))

    # %% initialize values for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(hyper.seed)
    criterion = nn.CrossEntropyLoss()

    # %%
    # 10-fold cross validation

    # setting
    foldperf = {}
    splits = KFold(n_splits=hyper.fold_num, shuffle=True, random_state=hyper.seed)

    # time
    time_start = time.time()
    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(train_dataset)))):

        print('Fold {}'.format(fold + 1))

        # sample the data for training and validation
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        # initialize loader
        train_loader = DataLoader(train_dataset, batch_size=hyper.batch_size, sampler=train_sampler,
                                  num_workers=hyper.num_workers)
        val_loader = DataLoader(train_dataset, batch_size=hyper.batch_size, sampler=val_sampler,
                                num_workers=int(hyper.num_workers / 4))

        # get model according to the channels after PCA and other hyper parameters
        model = MLP(input_channel, hyper.num_hidden_layer, hyper.num_channel)

        # load model to GPU/CPU
        model.to(device)

        # choose optimizer according to hyper.optimizer
        if hyper.optimizer == "adam":
            optimizer = optim.Adam(model.parameters(), lr=hyper.learning_rate)
        elif hyper.optimizer == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=hyper.learning_rate)

        # initialize data record dict
        history = {'epoch': [],
                   'train_loss': [], 'train_acc': [], 'train_precision': [], 'train_recall': [], 'train_f1': [],
                   'val_loss': [], 'val_acc': [], 'val_precision': [], 'val_recall': [], 'val_f1': [],
                   }

        for epoch in range(hyper.num_epochs):
            train_loss, train_acc, train_precision, train_recall, train_f1 = \
                train_epoch(model, device, train_loader, criterion, optimizer)
            val_loss, val_acc, val_precision, val_recall, val_f1 = \
                val_epoch(model, device, val_loader, criterion)
            print(
                "Epoch:{}/{} AVG Training Loss:{:.3f} AVG Val Loss:{:.3f} AVG Training Acc {:.2f} % AVG Val Acc {:.2f} %".format(
                    epoch,
                    hyper.num_epochs,
                    train_loss,
                    val_loss,
                    train_acc * 100,
                    val_acc * 100))

            # record
            history['epoch'].append(epoch)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['train_precision'].append(list(train_precision))
            history['train_recall'].append(list(train_recall))
            history['train_f1'].append(list(train_f1))

            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_precision'].append(list(val_precision))
            history['val_recall'].append(list(val_recall))
            history['val_f1'].append(list(val_f1))

        foldperf['fold{}'.format(fold + 1)] = history

    time_end = time.time()
    print('totally cost', time_end - time_start)
    json.dump(foldperf, open(Path(hyper.exp_path).joinpath("train_val.json"), "w"))

    # %%
    # full dataset training

    # initialize loader
    train_loader = DataLoader(train_dataset, batch_size=hyper.batch_size, num_workers=hyper.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=hyper.batch_size, num_workers=int(hyper.num_workers / 2))

    # get model according to the channels after PCA and other hyper parameters
    model = MLP(input_channel, hyper.num_hidden_layer, hyper.num_channel)

    # load model to GPU/CPU
    model.to(device)

    # choose optimizer according to hyper.optimizer
    if hyper.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=hyper.learning_rate)
    elif hyper.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=hyper.learning_rate)

    # initialize data record dict
    history_full = {'epoch': [],
                    'train_loss': [], 'train_acc': [], 'train_precision': [], 'train_recall': [], 'train_f1': [],
                    'test_loss': [], 'test_acc': [], 'test_precision': [], 'test_recall': [], 'test_f1': [],
                    }

    time_start = time.time()
    for epoch in range(hyper.num_epochs):
        # training
        train_loss, train_acc, train_precision, train_recall, train_f1 = \
            train_epoch(model, device, train_loader, criterion, optimizer)
        # testing
        test_loss, test_acc, test_precision, test_recall, test_f1 = \
            test_epoch(model, device, test_loader, criterion)

        # print info
        print(
            "Epoch:{}/{} AVG Training Loss:{:.3f} AVG test Loss:{:.3f} AVG Training Acc {:.2f} % AVG test Acc {:.2f} %".format(
                epoch,
                hyper.num_epochs,
                train_loss,
                test_loss,
                train_acc * 100,
                test_acc * 100))

        # record
        history_full['epoch'].append(epoch)
        history_full['train_loss'].append(train_loss)
        history_full['train_acc'].append(train_acc)
        history_full['train_precision'].append(list(train_precision))
        history_full['train_recall'].append(list(train_recall))
        history_full['train_f1'].append(list(train_f1))

        history_full['test_loss'].append(test_loss)
        history_full['test_acc'].append(test_acc)
        history_full['test_precision'].append(list(test_precision))
        history_full['test_recall'].append(list(test_recall))
        history_full['test_f1'].append(list(test_f1))

    # save the value
    json.dump(history_full, open(Path(hyper.exp_path).joinpath("train_test.json"), "w"))
    time_end = time.time()
    print('Total Time', time_end - time_start)

    # %%
    # calculate the useful value
    val_acc_epoch = np.array([foldperf['fold{}'.format(f)]['val_acc'] for f in range(1, hyper.fold_num + 1)])
    avg_val_acc = np.average(val_acc_epoch, axis=0)
    best_val_epoch = np.where(avg_val_acc == np.max(avg_val_acc))[0][0]
    best_test_epoch = np.where(history_full['test_acc'] == np.max(history_full['test_acc']))[0][0]
    # %%
    # calculate the useful value
    presentation = {
        'epoch': int(best_val_epoch),
        'test_acc': history_full['test_acc'][best_val_epoch],
        'test_f1': list(history_full['test_f1'][best_val_epoch]),
        'best_test_acc': history_full['test_acc'][best_test_epoch],
        'best_test_recall': list(history_full['test_recall'][best_test_epoch]),
        'best_test_precision': list(history_full['test_precision'][best_test_epoch]),
        'best_test_f1': list(history_full['test_f1'][best_test_epoch]),
        "val_acc": (np.max(avg_val_acc)),
        "val_f1": list(
            np.average(np.array([foldperf['fold{}'.format(f)]['val_f1'] for f in range(1, hyper.fold_num + 1)]),
                       axis=0)[best_val_epoch]),
        "best_train_acc":
            np.average(np.array([foldperf['fold{}'.format(f)]['train_acc'] for f in range(1, hyper.fold_num + 1)]),
                       axis=0)[best_val_epoch],
    }
    # %%
    # save the value
    json.dump(presentation, open(Path(hyper.exp_path).joinpath("pre.json"), "w"))


# %%
if __name__ == '__main__':
    # Example
    num_all_dim = 1024
    hypers = []
    hidden_layer = [1]
    num_channel = [2048]
    for h in hidden_layer:
        for c in num_channel:
            hypers.append(
                HyperClass(information_kept=1, exp_path_name="mlp_h{}_c{}".format(h, c), num_hidden_layer=h,
                           num_channel=c, num_epochs=5, base_path="./exp_t"))

    for h in hypers:
        h.save()
        main_func(h)
