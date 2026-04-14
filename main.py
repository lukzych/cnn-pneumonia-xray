import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import random

device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_EPOCHS = 13

#https://github.com/pheonix-18/Pneumonia-ChestXRay-Pytorch/blob/main/train.py
#https://github.com/mmanzanomo/pytorch-CNN-pneumonia-detection/tree/main


'''
Zmniejszenie datasetu żeby było po 50 na razie najlepszy wynik 
Teorytycznie im więcej tym lepiej ale jak augmentowałem obrazy z klasy NORMAL (np. Obracałem) tak aby
dorównać do liczby obrazów z klasy PNEUMONIA to wynik był gorszy.

Architektura modelu bardziej rozbudowana niż poprzednio i dla mniejszego datasetu lepiej działa
Dodatkowo w Transformerach dodałem więcej modyfikacji aby nie było Overfittingu nie wiem czy to ma 
znaczenie przy mniejszym datasecie

'''
class XRayModel(nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(input_shape, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.25)
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.25)
        )
        self.block_3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.25)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128*4*4, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, output_shape),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.classifier(x)
        return x


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


train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),      
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # wartości ImageNet
                         std=[0.229, 0.224, 0.225])
])


test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_datapath = r'./dataset/chest_xray/train'
test_datapath  = r'./dataset/chest_xray/test'

train_dataset = ImageFolder(train_datapath, transform=train_transform)
test_dataset  = ImageFolder(test_datapath,  transform=test_transform)

normal_inx    = [i for i, label in enumerate(train_dataset.targets) if label == 0]
pneumonia_inx = [i for i, label in enumerate(train_dataset.targets) if label == 1]

random.seed(42)
pneumonia_inx_balanced = random.sample(pneumonia_inx, len(normal_inx))

balanced_inx     = normal_inx + pneumonia_inx_balanced
balanced_dataset = Subset(train_dataset, balanced_inx)

train_dataloader = DataLoader(balanced_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader  = DataLoader(test_dataset,     batch_size=BATCH_SIZE, shuffle=False)

print(f"NORMAL: {len(normal_inx)} | PNEUMONIA: {len(pneumonia_inx_balanced)} | Razem: {len(balanced_inx)}")

torch.manual_seed(42)
model = XRayModel(input_shape=3, output_shape=1).to(device)  # input_shape=3 !
                                                              
loss_fn   = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0

    for images, labels in train_dataloader:
        images = images.to(device)
        labels = labels.float().to(device)

        predictions = model(images).squeeze()
        loss = loss_fn(predictions, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        correct += (predictions.round() == labels).sum().item()
        total += labels.size(0)

    print(f"Epoka {epoch+1}/{NUM_EPOCHS} | Loss: {epoch_loss/len(train_dataloader):.4f} | Accuracy: {correct/total*100:.2f}%")

model.eval()
correct = 0
total = 0
matrix = [[0, 0], [0, 0]]

with torch.no_grad():
    for images, labels in test_dataloader:
        images = images.to(device)
        labels = labels.float().to(device)

        predictions = model(images).squeeze()
        preds = predictions.round()

        preds_to_matrix = preds.cpu().int()
        labs_to_matrix  = labels.cpu().int()
        matrix = confusion_matrix(labs_to_matrix, preds_to_matrix, matrix)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {correct/total*100:.2f}%")
print(matrix)



'''
BEST SCORE
13 epok
Test Accuracy: 87.66%
[[197, 37], [40, 350]]
'''