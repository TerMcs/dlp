# import necessary packages
import os
import torch
import requests, zipfile, sys
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from download_fashion_mnist import download_fm
import torchvision
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

# Download fashion_MNIST from google drive and creating indices for training and validation set
root_dir = 'data'
download_fm(root_dir)

data_dir = os.path.join(root_dir, 'fashion_mnist_npy')

train_data = np.load(os.path.join(data_dir, 'train_data.npy'))
test_data = np.load(os.path.join(data_dir, 'test_data.npy'))
train_label = np.load(os.path.join(data_dir, 'train_labels.npy'))
test_label = np.load(os.path.join(data_dir, 'test_labels.npy'))

# split the training data to a new training data and validation data
class_indices = [[] for i in range(10)]
for i, v in enumerate(train_label):
    class_indices[v].append(i)

indices_all = np.random.permutation(len(class_indices[0]))
# take the first 1000 indices of indices_all
indices = indices_all[:1000]
# take the rest of the indices of indices_all
indices_rest = indices_all[1000:]

valid_indices = []
train_indices = []
for i in range(10):
    # build indices for validation set
    valid_indices.extend(np.array(class_indices[i])[indices])
    # TODO: build indices for training set (0.25 points)
    # your code here
    train_indices.extend(np.array(class_indices[i])[indices_rest])


# write the customer dataset based on the downloaded data and the indices
class FashionMnist(Dataset):
    """Fashion Mnist dataset"""

    def __init__(self, phase='train', transform=None):
        # download fashion_mnist data following Assignment2

        # TODO: now, split the predefined training data (1.5 points)
        if 'train' == phase:
            # your code here
            self.data = train_data[train_indices]
            self.label = train_label[train_indices]
        elif 'valid' == phase:
            # your code here
            self.data = train_data[valid_indices]
            self.label = train_label[valid_indices]

        elif 'test' == phase:
            # your code here
            self.data = test_data
            self.label = test_label

        else:
            assert True, 'wrong phase'

        self.transform = transform

        self.label_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        # TODO: get image and label according to the index (0.25 points)
        # your code here
        img = self.data[index]
        label = self.label[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            img = self.transform(img)

        return img, label

# checking training set
# randomly show some samples and their labels
# TODO: create a FashionMnist dataset for training, set transform to None (0.5 points)
# your code here
train_set =  FashionMnist(phase = 'train', transform = None)#this is the first 0.25 points, see the rest 0.25 points in Part 1.3

val_set =  FashionMnist(phase = 'valid', transform = None)#this is the first 0.25 points, see the rest 0.25 points in Part 1.3

test_set =  FashionMnist(phase = 'test', transform = None)#this is the first 0.25 points, see the rest 0.25 points in Part 1.3

# Set batch_size to 64, shuffling the training set. Number of workers here is set to 0. If your system is Linux,
# it is possible to try more workers to do multi-process parallel reading.
data_transform = transforms.Compose([transforms.ToTensor()])
train_set =  FashionMnist(phase = 'train', transform = True)#the other 0.25 points for create a FashionMnist dataset for training
train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0)

# TODO: create test_loader and valid_loader, both with no shuffling (1 points)
# your code here
val_set =  FashionMnist(phase = 'valid', transform = True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=0)
test_set =  FashionMnist(phase = 'test', transform = True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=0)