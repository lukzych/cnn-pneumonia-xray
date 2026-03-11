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
#https://www.learnpytorch.io/02_pytorch_classification/

device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
IMG_SIZE = (224,224)
NUM_EPOCHS = 5


class BaselineModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=224*224, out_features=64),
            nn.Linear(in_features=64, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layer_stack(x)
    
model = BaselineModel().to(device)

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
    transforms.Resize((IMG_SIZE)), #OBRAZ MUSI MIEĆ TEN SAM ROZMIAR TO JEST PROBLEM
    transforms.Grayscale(1),
    transforms.ToTensor()
])

train_datapath = r'./dataset/chest_xray/train'
train_dataset = ImageFolder(train_datapath, transform=data_trans)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_datapath = r'./dataset/chest_xray/test'
test_dataset = ImageFolder(test_datapath, transform=data_trans)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

val_datapath = r'./dataset/chest_xray/val'
val_dataset = ImageFolder(val_datapath,transform=data_trans)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


loss_fn = nn.BCELoss() #Binary Cross Entropy - do klasyfikacji binarnej
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) #model.parameters() zwraca wagi i biasy to co model ma się nauczyć


# Jedna epoka kończy się jak wszystkie obrazy zostały przetworzone,
#czyli 163 batche i tak dalej
for epoch in range(NUM_EPOCHS):

    model.train() #Tryb treningowy
    epoch_loss = 0 #zbiera sumę lossów z batchów, dzielomy przez liczbe batchy i jest średnia (linia 102)
    correct  = 0
    total = 0

    for images, labels in train_dataloader: #Pętla po batchach

        images = images.to(device)
        labels = labels.float().to(device) #BCELoss wymaga float

        #Forward pass
        #Obrazy lecą przez model (warstwy) i wychodzą predykcje
        #Potem licze jak bardzo predykjce różnią się od etykiet (loss)
        predictions = model(images).squeeze()
        loss = loss_fn(predictions, labels)   #labels - oczekiwany wynik, predictions - faktyczny wynik
        
        #zerujemy żeby gradienty się nie sumowały po batchach
        #wymazujemy gradienty dla starego
        optimizer.zero_grad()
        #obliczamy gradienty dla aktualnego batcha
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
matrix = [[0,0], [0,0]] #ta macierz pomyłek

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
print(matrix) #Zobaczymy


#Epoka 1/5 | Loss: 0.4076 | Accuracy: 85.18%
#Epoka 2/5 | Loss: 0.1675 | Accuracy: 93.75%
#Epoka 3/5 | Loss: 0.1657 | Accuracy: 93.65%
#Epoka 4/5 | Loss: 0.1465 | Accuracy: 94.44%
#Epoka 5/5 | Loss: 0.1427 | Accuracy: 95.03%
#Test Accuracy: 67.95%

'''Overfitting'''
#[[35, 199], [1, 389]] -> macierz pomyłek

                                #Co model 
                            #NORMAL   #PHNEUMONIA
#Labele     #NORMAL         35            199
            #PHNEUMONIA     1             389

#TODO jednolite zmienne dla notebooka i skryptu main.py bo jest syf
#TODO MACIERZ POMYŁEK (jako tako jest)
#TODO zrobić CNN do obrazów ta cała "konwolucja"
#TODO PRZEPISAC DO NOTEBOOKA