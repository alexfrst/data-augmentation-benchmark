import os
import sys
import zipfile
import config

import numpy as np
import torch
import wget
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms

sys.path.append('.')
from utils.print_utils import Symbols


# Load or download zipfile
def load_zipfile():
    zipfile_name = "dataset.zip"
    unzip_dir = "dataset"
    if os.path.exists(unzip_dir):
        print(f'Dataset already downloaded {Symbols.OK}')
    else:
        print(f'{unzip_dir} does not exist, downloading...')
        wget.download(config.zipfile_url, zipfile_name)
        print(f'Download complete {Symbols.OK}')
        print(f'Unzipping {zipfile_name}...')
        with zipfile.ZipFile(zipfile_name, 'r') as zip_ref:
            zip_ref.extractall(unzip_dir)
        print(f'Unzipping complete {Symbols.OK}')
        print(f'Removing {zipfile_name}...')
        os.remove(zipfile_name)
        print(f'Removing complete {Symbols.OK}')
    return unzip_dir


def check_labels_corectness(dataset):
    labels = np.array([x[1] for x in dataset])

    nb_classes = labels.max() + 1

    assert labels.min() == 0, f"Labels must start at 0 {Symbols.FAIL}"
    assert len(np.unique(labels)) == nb_classes, f"Labels must be in range(0, nb_classes) {Symbols.FAIL}"

    return nb_classes


def load_dataset(additional_transforms=()):
    print("Loading dataset...")

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    train_image_directory = "dataset/dataset-train"
    test_image_directory = "dataset/dataset-test"

    train_transforms = transforms.Compose([
        transforms.Resize([299, 299]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    test_transforms = transforms.Compose([
        transforms.Resize([299, 299]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    np.random.seed(42)

    dataset_train_val = datasets.ImageFolder(train_image_directory, train_transforms)
    dataset_test = datasets.ImageFolder(test_image_directory, test_transforms)

    samples_train, samples_val = train_test_split(dataset_train_val.samples, test_size=0.1)

    print("Nombre d'images de train : %i" % len(samples_train))
    print("Nombre d'images de val : %i" % len(samples_val))
    print("Nombre d'images de test : %i" % len(dataset_test))

    # on définit les datasets et loaders pytorch à partir des listes d'images de train / val / test
    dataset_train = datasets.ImageFolder(train_image_directory, train_transforms)
    dataset_train.samples = samples_train
    dataset_train.imgs = samples_train

    dataset_val = datasets.ImageFolder(train_image_directory, train_transforms)
    dataset_val.samples = samples_val
    dataset_val.imgs = samples_val

    nb_classes_train = check_labels_corectness(samples_train)
    nb_classes_val = check_labels_corectness(samples_val)
    nb_classes_test = check_labels_corectness(dataset_test)

    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=16, shuffle=True, num_workers=4)
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=16, shuffle=True, num_workers=4)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=16, shuffle=True, num_workers=4)

    assert nb_classes_train == nb_classes_val == nb_classes_test, f"Number of classes in train test and val must be the same {Symbols.FAIL}"

    print(f"Apprentissage sur {nb_classes_train} classes")
    print(f"Data loading and integrity check done {Symbols.OK}")

    return loader_train, loader_val, loader_test, nb_classes_train


if __name__ == '__main__':
    load_zipfile()
    load_dataset()
