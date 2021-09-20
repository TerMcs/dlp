import pandas as pd
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets

data_dir = 'datasets/'
filename = 'metadata.csv'

csv_file_path = data_dir + filename
region = 'L-SPINE' # Should be a string based on format of DICOM filenames.
plane = 'TSE_SAG__' # Other options: TSE_TRA__, T1_TSE_SAG__, T2_TSE_SAG__
slices = ["004.ima",
          "005.ima",
          "006.ima",
          "007.ima",
          "008.ima",
          "009.ima",
          "010.ima",
          "011.ima",
          "012.ima"]

def get_file_paths(csv_file_path, region, plane, slices):

    # read in the csv file containing all DICOM metadata:
    data = pd.read_csv(csv_file_path)

    # Select relevant columns:
    df = pd.DataFrame(data, columns=["PatientAge", "PathToFolder", "FileName"])

    # Need to add in a slash here so that the full path is correct in the end after concatenation:
    df['FileName'] = '/' + df['FileName'].astype(str)
    df["FullPathDICOM"] = df["PathToFolder"] + df["FileName"]
    df = df.dropna(subset=['PatientAge'])

    # Select region based on string:
    df = df[df.FullPathDICOM.str.contains(region)]

    # Select plane (and T1 or T2 or both)
    df = df[df.FullPathDICOM.str.contains(plane)]

    # Select slices based on list of strings provided:
    df = df[df.FullPathDICOM.str.contains('|'.join(slices))]

    # Clean up the patient age if needed and convert to integer:
    df['PatientAge'] = df['PatientAge'].str.extract('(\d+)', expand=False).astype(int)
    df = pd.DataFrame(df, columns=["PatientAge", "FullPathDICOM"])
    df['FullPathNPY'] = df['FullPathDICOM'].str.replace(".ima", ".npy")

    return df

class LumbarSpineDicomDataset(torch.utils.data.Dataset):
  def __init__(self, list_IDs, labels, transform=None):
        self.labels = labels
        self.list_IDs = list_IDs
        self.transform = transform

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        img_path = data_dir + ID
        image = np.load(img_path, allow_pickle=True)
        label = self.labels[ID]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_lumbar_mri():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize(320),
                                    transforms.RandomHorizontalFlip(p=0.2),
                                    torchvision.transforms.RandomVerticalFlip(p=0.2),
                                    # transforms.RandomCrop(200),
                                    torchvision.transforms.RandomRotation(5),
                                    transforms.Normalize(0.5, 0.5, inplace=False)
                                    ])

    df = get_file_paths(csv_file_path, region, plane, slices)

    labels = pd.Series(df.PatientAge.values, index=df.FullPathNPY).to_dict()

    list_IDs = list(df["FullPathNPY"])

    dataset = LumbarSpineDicomDataset(list_IDs, labels, transform=transform)

    train_set, val_set = torch.utils.data.random_split(dataset, [8726, 1000])
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=32, shuffle=False)

    return train_loader, val_loader

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
