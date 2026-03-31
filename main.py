import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import cv2

import random
#https://www.learnpytorch.io/03_pytorch_computer_vision/
#https://www.learnpytorch.io/02_pytorch_classification/
#https://www.youtube.com/watch?v=f3g1zGdxptI&list=PLoROMvodv4rOmsNzYBMe0gJY2XS8AQg16&index=5

device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
IMG_SIZE = (224,224)
NUM_EPOCHS = 5


class BaselineModel(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=3, # filtr
                      stride=1, # default
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2) # default stride value is same as kernel_size
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            
            nn.Linear(in_features=hidden_units*56*56, 
                      out_features=output_shape),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.classifier(x)
        return x

torch.manual_seed(42)
model = BaselineModel(
    input_shape = 1,
    hidden_units = 32,
    output_shape = 1
).to(device)

#NORMAL - 0, PHNE - 1
def confusion_matrix(actual, predicted, matrix):

    for actual, predicted in zip(actual, predicted):
        if predicted == 1 and actual == 1:
            matrix[1][1] += 1

        if predicted == 0 and actual == 1:
            matrix[1][0] += 1

        if predicted == 0 and actual == 0:
            matrix[0][0] += 1
        
        if predicted == 1 and actual == 0:
            matrix[0][1] += 1

    return matrix


data_trans = transforms.Compose([
    transforms.Resize((IMG_SIZE)),
    transforms.Grayscale(1),
    transforms.ToTensor()
])

train_trans = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.Grayscale(1),
    transforms.RandomHorizontalFlip(),    
    transforms.RandomRotation(10),        
    transforms.ToTensor()
])


train_datapath = r'./dataset/chest_xray/train'
train_dataset = ImageFolder(train_datapath, transform=data_trans)
#train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_datapath = r'./dataset/chest_xray/test'
test_dataset = ImageFolder(test_datapath, transform=data_trans)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

val_datapath = r'./dataset/chest_xray/val'
val_dataset = ImageFolder(val_datapath,transform=data_trans)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

#Zrównoważenie datasetu dodanie zmodyfokowanych obrazów normal (obróconych i lustrzane odbicie)
normal_inx = []

for i, label in enumerate(train_dataset.targets):
    if label == 0:
        normal_inx.append(i)

normal_dataset = Subset(train_dataset, normal_inx)

normal_dataset_aug = Subset(
    ImageFolder(train_datapath, transform=train_trans),  # transformer z augmentacją
    normal_inx
)

train_dataset_final = ConcatDataset([
    train_dataset,        # wszystkie oryginalne
    normal_dataset_aug,    # tylko NORMAL z augmentacją
    normal_dataset_aug
])

train_dataloader = DataLoader(train_dataset_final, batch_size=BATCH_SIZE, shuffle=True)

loss_fn = nn.BCELoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) #model.parameters() zwraca wagi i biasy to co model ma się nauczyć

counter = Counter(train_dataset_final.datasets[0].targets + 
                  [train_dataset_final.datasets[1].dataset.targets[i] for i in normal_inx])

# albo prościej - po prostu zlicz ręcznie
normal_count = len(normal_inx) * 2      # oryginalne + augmentowane
pneumonia_count = len(train_dataset.targets) - len(normal_inx)

print(f"NORMAL: {normal_count}")
print(f"PNEUMONIA: {pneumonia_count}")
print(f"Razem: {normal_count + pneumonia_count}")


for epoch in range(NUM_EPOCHS):

    model.train() 
    epoch_loss = 0 
    correct  = 0
    total = 0

    for images, labels in train_dataloader: #Pętla po batchach

        images = images.to(device)
        labels = labels.float().to(device) #BCELoss wymaga float

        predictions = model(images).squeeze()
        loss = loss_fn(predictions, labels)   
        
        
        optimizer.zero_grad()
        
        loss.backward()

        #updatuje wagi
        optimizer.step()

        epoch_loss = epoch_loss + loss.item() #.item() chcemy dodać float do float a loss to tensor
        correct += (predictions.round() == labels).sum().item() #Próg 0.5 (round)
        total = total + labels.size(0)

    print(f"Epoka {epoch+1}/{NUM_EPOCHS} | Loss: {epoch_loss/len(train_dataloader):.4f} | Accuracy: {correct/total*100:.2f}%")


#Testowanie
model.eval() #Tryb ewaluacji - sprawdzanie
correct = 0
total = 0
matrix = [[0,0], [0,0]]

with torch.no_grad():
    for images, labels in test_dataloader:
        images = images.to(device)
        labels = labels.float().to(device)

        predictions = model(images).squeeze()

        preds_to_matrix = predictions.round().cpu().int()
        labs_to_matrix = labels.cpu().int()
        matrix = confusion_matrix(labs_to_matrix, preds_to_matrix, matrix)

        correct += (predictions.round() == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {correct/total*100:.2f}%")
print(matrix)


'''

LATEST SCORE 
NORMAL: 2682 
PNEUMONIA: 3875
Razem: 6557

Epoka 1/5 | Loss: 0.3406 | Accuracy: 82.25%
Epoka 2/5 | Loss: 0.1045 | Accuracy: 96.30%
Epoka 3/5 | Loss: 0.0820 | Accuracy: 97.06%
Epoka 4/5 | Loss: 0.0707 | Accuracy: 97.61%
Epoka 5/5 | Loss: 0.0627 | Accuracy: 97.76%
Test Accuracy: 73.72%
[[73, 161], [3, 387]]
'''