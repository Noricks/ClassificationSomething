from models import net_I, net_A
import torchvision
from torch import nn, optim
from utils.hyper_class import HyperClass
# %%
def get_model(name):
    class_num = 10
    num_in_channels = 1
    if name == "I":
        model = net_I.get_net(class_num, num_in_channels)
    elif name == "A":
        model = net_A.get_net(class_num, num_in_channels)
    elif name == "resnet-18":
        model = torchvision.models.resnet18(pretrained=False, num_classes=class_num)
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
