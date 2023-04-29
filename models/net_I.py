import torch.nn as nn
import torch.nn.functional as F


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
        # short cut bypass information
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=(1, 1), stride=(stride, stride), bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
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
        self.layer1 = self.make_layer(NetBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(NetBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(NetBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(NetBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        # assemble the network
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)  # flatten the 2D matrix
        out = self.fc(out)
        return out


def get_net(num_classes=10, num_in_channel=1):
    return Net(Block, num_classes, num_in_channel)
