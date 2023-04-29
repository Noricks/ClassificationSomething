import torch.nn as nn
import torch.nn.functional as F


# basic block of the network
class Block(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(Block, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=(3, 3), stride=(stride, stride), padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )

    def forward(self, x):
        out = self.left(x)
        out = F.relu(out)
        return out


class Net(nn.Module):
    """
        Ref: torchvision.models.resnet
        Ref: https://github.com/townblack/pytorch-cifar10-resnet18
    """

    def __init__(self, NetBlock, num_classes=10, num_in_channel=1):
        super(Net, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_in_channel, 64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(NetBlock, 64, 1, stride=2)
        self.fc = nn.Linear(1024, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)  # flatten the 2D matrix
        out = self.fc(out)  # full connection
        return out


def get_net(num_classes=10, num_in_channel=1):
    return Net(Block, num_classes, num_in_channel)
