import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

# from train import batch_size as bs

import time
import os
import PIL.Image as Image
from IPython.display import display
import numpy as np

dataset_dir = "../data/Rice_Image_Dataset/"

train_transform = transforms.Compose([
transforms.RandomHorizontalFlip(),
transforms.RandomVerticalFlip(),
transforms.RandomRotation(90),
transforms.ColorJitter(brightness=0.2, contrast=0.2, hue=0.2),
transforms.RandomGrayscale(0.2),
transforms.RandomCrop(224),
transforms.ToTensor(),
transforms.Normalize((0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010)),
])
val_test_transform = transforms.Compose([
transforms.RandomCrop(224),
transforms.ToTensor(),
transforms.Normalize((0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010)),
])

bs = 32

train_dataset = torchvision.datasets.ImageFolder(root=dataset_dir+"train", transform = train_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = bs, shuffle=True, num_workers = 2)

val_dataset = torchvision.datasets.ImageFolder(root=dataset_dir+"val", transform = val_test_transform)
valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size = bs, shuffle=False, num_workers = 2) 

test_dataset = torchvision.datasets.ImageFolder(root=dataset_dir+"test", transform = val_test_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = bs, shuffle=False, num_workers = 2)

print("Data Loader Successed!")