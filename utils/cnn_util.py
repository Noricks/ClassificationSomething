from models import net_I, net_A
import torchvision
from torch import nn, optim
from utils.hyper_class import HyperClass
from torchvision import transforms
from datasets.CIFAR import CIFAR
from datasets.Mushroom import Mushroom


# %%
def get_model(hyper: HyperClass):
    name = hyper.network_name
    class_num = hyper.class_num
    num_in_channels = 3
    if name == "I":
        model = net_I.get_net(class_num, num_in_channels)
    elif name == "A":
        model = net_A.get_net(class_num, num_in_channels)
    elif name == "resnet-18":
        model = torchvision.models.resnet18(weights=True)
        model.fc = nn.Linear(512, class_num)
    elif name == "resnet-34":
        model = torchvision.models.resnet34(weights=True)
        model.fc = nn.Linear(512, class_num)
    elif name == "resnet-50":
        model = torchvision.models.resnet50(weights=True)
        model.fc = nn.Linear(2048, class_num)
    elif name == "efficientnet-b0":
        model = torchvision.models.efficientnet_b0(weights=True, num_classes=class_num)
    else:
        # Other Networks could be modified from these two networks
        # To make it simple they are not list here
        raise Exception("Can not find network")

    return model


def get_optimizer(hyper: HyperClass, model: nn.Module):
    # choose optimizer according to hyper.optimizer
    if hyper.optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=hyper.learning_rate)
    elif hyper.optimizer == "adadelta":
        optimizer = optim.Adadelta(model.parameters(), lr=hyper.learning_rate)
    elif hyper.optimizer == "adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=hyper.learning_rate)
    elif hyper.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=hyper.learning_rate)
    else:
        raise ValueError("Optimizer not found")
    return optimizer


def get_criterion(hyper: HyperClass):
    return nn.CrossEntropyLoss()


def get_dataset(hyper: HyperClass):
    if hyper.dataset == "CIFAR-10":
        return CIFAR().get_dataset(hyper)
    elif hyper.dataset == "CIFAR-100":
        return CIFAR().get_dataset(hyper)
    elif hyper.dataset == "Mushroom":
        return Mushroom().get_dataset(hyper)
    else:
        raise ValueError("Dataset not found")
