# %%
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pathlib import Path
import json
import os
from utils.cnn_util import get_model
from utils.epoch_utils import train_epoch, test_epoch


# %% hyper
class HyperClass(object):
    r"""The class to store and handle all the hyper parameters

    Should be sent into main_func
    """

    def __init__(self,
                 seed=20124861,
                 num_epochs=100,
                 batch_size=128,
                 learning_rate=0.01,
                 num_workers=4,
                 optimizer="sgd",
                 exp_path_name="cnn2",
                 base_path="/mnt/emc01/zeyu/mlcw/exp/",
                 network_name="I"
                 ):
        self.network_name = network_name
        self.seed = seed
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_workers = num_workers
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

    # %% initialize values for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(hyper.seed)
    criterion = nn.CrossEntropyLoss()

    # %%
    # initialize loader
    train_loader = DataLoader(train_dataset, batch_size=hyper.batch_size, num_workers=hyper.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=hyper.batch_size, num_workers=int(hyper.num_workers / 2))

    # get model from the name
    model = get_model(hyper.network_name)

    # load model to GPU/CPU
    model.to(device)

    # choose optimizer according to hyper.optimizer
    if hyper.optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=hyper.learning_rate)
    elif hyper.optimizer == "adadelta":
        optimizer = optim.Adadelta(model.parameters(), lr=hyper.learning_rate)
    elif hyper.optimizer == "adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=hyper.learning_rate)
    elif hyper.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=hyper.learning_rate,
                              momentum=0.9, weight_decay=5e-4)
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

        # record all the values
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
    # get the epoch of best testing accuracy
    best_test_epoch = np.where(history_full['test_acc'] == np.max(history_full['test_acc']))[0][0]
    # %%
    # record data
    presentation = {
        'epoch': int(best_test_epoch),
        'train_acc': history_full['train_acc'][best_test_epoch],
        'test_acc': history_full['test_acc'][best_test_epoch],
        'test_precision': list(history_full['test_precision'][best_test_epoch]),
        'test_recall': list(history_full['test_recall'][best_test_epoch]),
        'test_f1': list(history_full['test_f1'][best_test_epoch]),
    }
    # %%
    # save the recorded data to file
    json.dump(presentation, open(Path(hyper.exp_path).joinpath("pre.json"), "w"))


# %%
if __name__ == '__main__':
    # Example
    optimizers = ["adamw"]
    lrs = [0.01]
    hypers = []
    for o in optimizers:
        for l in lrs:
            hypers.append(
                HyperClass(optimizer=o, learning_rate=l, exp_path_name="cnn_{}_l{}".format(o, l), num_epochs=5,
                           base_path="./exp_t", network_name="I"))

    for h in hypers:
        h.save()
        main_func(h)
