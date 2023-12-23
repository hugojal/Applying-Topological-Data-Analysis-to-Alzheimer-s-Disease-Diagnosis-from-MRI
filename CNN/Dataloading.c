!pip install matplotlib-venn

!apt-get -qq install -y libfluidsynth1

!pip install libarchive

!apt-get -qq install -y graphviz && pip install pydot
import pydot

!pip install cartopy
import cartopy

import torch
import numpy as np
import pandas as pd
from torch import nn
from time import time
from os import path
from torchvision import transforms
import random
from copy import deepcopy
import urllib.request
import ssl
import tarfile
import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import os
from os import path
import glob

class MRIDataset(torch.utils.data.Dataset):
    """
    MRI dataset
    """

    def __init__(self, img_dir, df, transform=None):
        self.img_dir = img_dir
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.df['image_id'][idx])
        img = np.load(img_path)
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=0)
        img = torch.from_numpy(img)
        label = self.df['label'][idx]
        label = torch.tensor(label)
        if self.transform:
            img = self.transform(img)
        return img, label


class CustomNetwork(nn.Module):
    """
    Custom network
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3


gcontext = ssl.SSLContext()
dataseturl = "https://aramislab.paris.inria.fr/files/data/databases/DL4MI/OASIS-1-dataset_pt_new.tar.gz"
fstream = urllib.request.urlopen(dataseturl, context=gcontext)
tarfile = tarfile.open(fileobj=fstream, mode="r:gz")
tarfile.extractall()


OASIS_df = pd.read_csv('OASIS-1_dataset/tsv_files/lab_1/OASIS_BIDS.tsv', sep='\t')
print(OASIS_df.head())
_ = OASIS_df.hist(figsize=(20, 14))

