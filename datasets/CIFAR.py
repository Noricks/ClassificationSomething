from utils.hyper_class import HyperClass
from torchvision import transforms
import torchvision

class CIFAR:
    def __init__(self):
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5)
            ])

    def get_dataset(self, hyper: HyperClass):
        if hyper.dataset == "CIFAR-10":
            train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                         download=True, transform=self.transform)

            test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                        download=True, transform=self.transform)
        elif hyper.dataset == "CIFAR-100":
            train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                                          download=True, transform=self.transform)
            test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                                         download=True, transform=self.transform)
        else:
            raise ValueError("Dataset not found")

        return train_dataset, test_dataset
