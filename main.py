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

device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32

IMG_SIZE = (224,224)

data_trans = transforms.Compose([
    transforms.Resize((IMG_SIZE)), #OBRAZ MUSI MIEĆ TEN SAM ROZMIAR TO JEST PROBLEM
    transforms.Grayscale(1),
    transforms.ToTensor()
])


test_datapath = r'./dataset/chest_xray/test'
train_datapath = r'./dataset/chest_xray/train'
val_datapath = r'./dataset/chest_xray/val'

test_dataset = ImageFolder(test_datapath, transform=data_trans)
train_dataset = ImageFolder(train_datapath, transform=data_trans)
val_dataset = ImageFolder(val_datapath,transform=data_trans)



test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)


class BaselineModel(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )

    def forward(self, x):
        return torch.sigmoid(self.layer_stack(x))
    
model = BaselineModel(
    input_shape = 224*224,
    hidden_units=64,
    output_shape=1
).to(device)



#Training
learning_rate = 0.01
NUM_EPOCHS = 5

loss_fn = nn.BCELoss() #Binary Cross Entropy - do klasyfikacji binarnej (out: sigmoid)
#MSE (Mean Squared Entropy - do regresji -> liczba ciągła)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) 
#model.parameters() zwraca wagi i biasy to co model ma się nauczyć


# Jedna epoka kończy się jak wszystkie obrazy zostały przetworzone,
#czyli 163 batche i tak dalej
for epoch in range(NUM_EPOCHS):
    model.train() #Tryb treningowy
    epoch_loss = 0
    correct  = 0
    total = 0

    for images, labels in train_dataloader: #Pętla po batchach

        images = images.to(device)
        labels = labels.float().to(device) #BCELoss wymaga float

        #Forward pass
        #Obrazy lecą przez model (warstwy) i wychodzą predykcje
        #Potem licze jak bardzo predykjce różnią się od etykiet (loss)
        predictions = model(images).squeeze()
        loss = loss_fn(predictions, labels)
        
        #zerujemy żeby gradienty się nie sumowały po batchach
        #wymazujemy gradienty z batcha 1
        optimizer.zero_grad()
        #obliczamy gradienty dla batcha 2
        loss.backward()

        #updatuje wagi używając gradientów batcha 2
        #czyli w którą strone zmienić wagi
        optimizer.step()

        epoch_loss = epoch_loss + loss.item()
        correct += (predictions.round() == labels).sum().item()
        total = total + labels.size(0)

    print(f"Epoka {epoch+1}/{NUM_EPOCHS} | Loss: {epoch_loss/len(train_dataloader):.4f} | Accuracy: {correct/total*100:.2f}%")


#Testowanie
model.eval() #Tryb ewaluacji - sprawdzanie
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_dataloader:
        images = images.to(device)
        labels = labels.float().to(device)

        predictions = model(images).squeeze()
        correct += (predictions.round() == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {correct/total*100:.2f}%")

#Epoka 1/5 | Loss: 0.3799 | Accuracy: 86.27%
#Epoka 2/5 | Loss: 0.1782 | Accuracy: 93.08%
#Epoka 3/5 | Loss: 0.1690 | Accuracy: 93.37%
#Epoka 4/5 | Loss: 0.1397 | Accuracy: 94.57%
#Epoka 5/5 | Loss: 0.1531 | Accuracy: 93.98%
#Test Accuracy: 70.99%


#Flatten + Linear słabe rozwiązanie (do przewidzenia w sumie)
#TODO zrobić CNN do obrazów 

'''
WIKIPEDIA
Nadmierne dopasowanie (ang. overfitting) a. przeuczenie 
(branż. „przetrenowanie”, ang. overtraining) efekt obserwowany np. w statystyce,
gdy model statystyczny ma zbyt dużo parametrów w stosunku do rozmiaru próby,
na podstawie której był konstruowany; 
w przypadku uczenia maszynowego modele o dużej złożoności mogą świetnie dopasować 
się do danych uczących, jednak będą dawały słabe wyniki, 
gdy zastosuje się je do danych, z którymi nie zetknęły się podczas uczenia.

'''