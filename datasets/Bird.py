import os

from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms

from utils.hyper_class import HyperClass

ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
from tqdm import tqdm


class Bird(Dataset):
    def __init__(self):
        self.train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5),
                transforms.Resize((224, 224)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
            ])
        self.test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5),
                transforms.Resize((224, 224)),
            ])

    def get_dataset(self, hyper: HyperClass):
        if hyper.dataset == "Bird":
            train_dataset = BirdDataSet(root='/home/bird/train/', train=True,
                                            download=True, transform=self.train_transform)

            test_dataset = BirdDataSet(root='/home/bird/test/', train=False,
                                           download=True, transform=self.test_transform)
        else:
            raise ValueError("Dataset not found")

        return train_dataset, test_dataset


class BirdDataSet(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.download = download
        self.classes = list(self.load_data_labels())
        self.classes.sort()
        self.classes_to_idx = {self.classes[i]: i for i in range(len(self.classes))}

        # self.target_url = "https://www.kaggle.com/datasets/maysee/mushrooms-classification-common-genuss-images" \
        #                   "/download?datasetVersionNumber=1"

        if self.train:
            self.train_data, self.train_labels = self.load_data()
        else:
            self.test_data, self.test_labels = self.load_data()

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def load_data_labels(self):
        folder_names = os.listdir(self.root)
        return folder_names

    def load_data_dir(self):
        # check if data is downloaded
        if not os.path.exists(self.root):
            raise ValueError("Dataset not downloaded")

        # read all the folder names
        folder_names = os.listdir(self.root)

        # read all the images and labels
        images = []
        labels = []
        for folder_name in folder_names:
            folder_path = os.path.join(self.root, folder_name)
            images_in_folder = os.listdir(folder_path)
            for image_name in images_in_folder:
                image_path = os.path.join(folder_path, image_name)
                images.append(image_path)
                labels.append(folder_name)

        # return images and labels
        return images, labels

    def load_data(self):
        images, labels = self.load_data_dir()
        # read all the images from file name to numpy array using Image
        data = []
        print("Loading data")
        for image_path in tqdm(images):
            image = Image.open(image_path)
            image = image.convert('RGB')
            image = np.array(image)
            data.append(image)
        # convert the labels to index
        labels = [self.classes_to_idx[label] for label in labels]
        labels = np.array(labels)
        return data, labels


if __name__ == '__main__':
    dataset = BirdDataSet(root='/home/bird/train/', download=True)
