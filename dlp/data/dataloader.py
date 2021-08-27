import os
import importlib
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets

def get_mnist(batch_size):
    train_dataset = datasets.MNIST(root='../../data',
                                   train=True,
                                   transform=transforms.ToTensor(),
                                   download=True
                                   )

    test_dataset = datasets.MNIST(root='../../data',
                                  train=False,
                                  transform=transforms.ToTensor(),
                                  download=True
                                  )

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True
                              )

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False
                             )

    return train_loader, test_loader


def get_cifar10(batch_size):
    transform = transforms.Compose([transforms.Pad(4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(32),
                                    transforms.ToTensor()
                                    ])

    train_dataset = torchvision.datasets.CIFAR10(root='../../data',
                                                 train=True,
                                                 transform=transform,
                                                 download=True
                                                 )

    test_dataset = torchvision.datasets.CIFAR10(root='../../data',
                                                train=False,
                                                transform=transforms.ToTensor()
                                                )

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True
                                               )

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False
                                              )
    return train_loader, test_loader
