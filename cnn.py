# %%
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.cnn_util import get_model, get_optimizer, get_criterion, get_dataset
from utils.epoch_utils import train_epoch, test_epoch
from utils.hyper_class import HyperClass


# %%
def main_func(hyper: HyperClass):
    torch.manual_seed(hyper.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # %% load dataset
    train_dataset, test_dataset = get_dataset(hyper)

    # %% initialize values for training
    criterion = get_criterion(hyper)

    # %%
    # initialize loader
    train_loader = DataLoader(train_dataset, batch_size=hyper.batch_size, num_workers=hyper.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=hyper.batch_size, num_workers=int(hyper.num_workers / 2))

    # get model from the name
    model = get_model(hyper)

    # load model to GPU/CPU
    model.to(device)

    # choose optimizer according to hyper.optimizer
    optimizer = get_optimizer(hyper, model)

    # initialize data record dict
    history_full = {'epoch': [],
                    'train_loss': [], 'train_acc': [], 'train_precision': [], 'train_recall': [], 'train_f1': [],
                    'test_loss': [], 'test_acc': [], 'test_precision': [], 'test_recall': [], 'test_f1': [],
                    }

    time_start = time.time()
    for epoch in tqdm(range(hyper.num_epochs)):
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
                           base_path="./exp_t", network_name="mixer_b_16", batch_size=32, num_workers=4, ))

    for h in hypers:
        h.save()
        main_func(h)
