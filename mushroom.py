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
    train_loader = DataLoader(train_dataset, batch_size=hyper.batch_size, num_workers=hyper.num_workers, shuffle=True)
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
        train_loss, train_acc = train_epoch(model, device, train_loader, criterion, optimizer)
        # testing
        test_loss, test_acc = test_epoch(model, device, test_loader, criterion)

        # print info
        print(
            "Epoch:{}/{} Train Loss: {:.3f} Test Loss: {:.3f} Train Acc {:.2f} % Test Acc {:.2f} %".format(
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

        history_full['test_loss'].append(test_loss)
        history_full['test_acc'].append(test_acc)
    # save the value
    json.dump(history_full, open(Path(hyper.exp_path).joinpath("train_test.json"), "w"))
    time_end = time.time()
    time_period = time_end - time_start
    # time to h-m-s
    m, s = divmod(time_period, 60)
    h, m = divmod(m, 60)
    print('Total Time {} h {} m {:.2f} s'.format(h, m, s))

    # %%
    # get the epoch of best testing accuracy
    best_test_epoch = np.where(history_full['test_acc'] == np.max(history_full['test_acc']))[0][0]
    # %%
    # record data
    presentation = {
        'epoch': int(best_test_epoch),
        'train_acc': history_full['train_acc'][best_test_epoch],
        'test_acc': history_full['test_acc'][best_test_epoch],
    }
    # %%
    # save the recorded data to file
    json.dump(presentation, open(Path(hyper.exp_path).joinpath("pre.json"), "w"))

    return presentation
# %%
if __name__ == '__main__':
    # Example
    optimizers = ["adamw"]
    lrs = [1e-3]
    hypers = []
    for o in optimizers:
        for l in lrs:
            hypers.append(
                HyperClass(optimizer=o, learning_rate=l, exp_path_name="cnn_{}_l{}".format(o, l), num_epochs=50,
                           base_path="./exp_t", network_name="vit_b_16", batch_size=32, num_workers=4, dataset="Mushroom"))

    for h in hypers:
        h.save()
        main_func(h)

    # import optuna
    #
    # def objective(trial):
    #     l = trial.suggest_float('learning_rate', 1e-5, 1e-1)
    #     o = trial.suggest_categorical('optimizer', ["adamw", "sgd"])
    #     b = trial.suggest_int('batch_size', 16, 64)
    #     n = trial.suggest_categorical('network_name', ["resnet-18", "resnet-34", "resnet-50"])
    #     h = HyperClass(optimizer=o, learning_rate=l, exp_path_name="cnn_{}_l{}_b{}_n{}".format(o, l, b, n), num_epochs=50,
    #                       base_path="./exp_t", network_name=n, batch_size=b, num_workers=4, dataset="Mushroom")
    #     h.save()
    #     presentation = main_func(h)
    #     return presentation['test_acc']
    #
    # study = optuna.create_study(direction='maximize')
    # study.optimize(objective, n_trials=2)
