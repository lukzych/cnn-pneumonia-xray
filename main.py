import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

import numpy as np
import matplotlib.pyplot as plt

import cv2

#https://www.learnpytorch.io/03_pytorch_computer_vision/

data_trans = transforms.Compose([
    transforms.Resize((512,512)), #OBRAZ MUSI MIEĆ TEN SAM ROZMIAR TO JEST PROBLEM
    transforms.ToTensor()
])


test_datapath = r'./dataset/chest_xray/test'
train_datapath = r'./dataset/chest_xray/train'
val_datapath = r'./dataset/chest_xray/val'

test_dataset = ImageFolder(test_datapath, transform=data_trans)
train_dataset = ImageFolder(train_datapath, transform=data_trans)
val_dataset = ImageFolder(val_datapath,transform=data_trans)

BATCH_SIZE = 32

#BATCH_SIZE
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)


#['NORMAL', 'PHNEUMONIA']
class_names_train = train_dataset.classes
print(class_names_train)


print(f"DataLoaders: {train_dataloader, test_dataloader}")
print(f"Length of train dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}")
print(f"Length of test dataloader: {len(test_dataloader)} batches of {BATCH_SIZE}")

#iter() zmienia train_dataloader na iterator, a next() pobiera pierwszy batch danych
#features - tensor z obrazami(batch_size, kanały, wysokość, szerokość)
#labels - tensor z etykietami (batch_size)
train_features_batch, train_labels_batch = next(iter(train_dataloader))
print(train_features_batch.shape, train_labels_batch.shape)


random_inx = torch.randint(0, len(train_features_batch), size=[1]).item() #losowy tensor z 1 liczbą np. tensor([2])   .item() # zamienia tensor na zwykłą liczbę całkowitą -> np. 2 

#Dokumentacja ImageFolder
#krotka (tuple)
img, label = train_features_batch[random_inx], train_labels_batch[random_inx]

plt.imshow(img.permute(1,2,0)) #Matplotlib oczekuje formatu(H, W, C) -> (512, 512, 3) tensor ma format (C, H, W) (3, 512, 512) permute i elo
plt.title(class_names_train[label])
plt.axis("Off")
print(f"Image size: {img.shape}" )
print(f"Label: {label}, label size: {label.size}")
plt.show()

#CO DALEJ / PAMIETAC / NAUCZYC
#TODO Co z tym Resize
#TODO BASELINE MODEL
#TODO nn.Sequential -> nn.Linear
#TODO nn.Flatten() -> co to
#TODO forward
#TODO Moze Jupyter

