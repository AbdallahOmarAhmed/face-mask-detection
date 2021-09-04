import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import os
from torchvision.io import read_image
from face_mask_DataLoader import faceMaskAccSet, faceMaskTestSet, faceMaskDataSet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#load the model
model = torchvision.models.resnet18()
n = model.fc.in_features
model.fc = nn.Linear(n, 1)
model.load_state_dict(torch.load("weights/modelTR.pth"))
model = model.to(device)
model.eval()

data_train = faceMaskDataSet()
dataloader_train = DataLoader(data_train, batch_size=1,\
 shuffle=False, num_workers=6)

data_test = faceMaskTestSet()
dataloader_test = DataLoader(data_test, batch_size=1,\
 shuffle=False, num_workers=6)

data_val = faceMaskAccSet()
dataloader_val = DataLoader(data_val, batch_size=1,\
 shuffle=False, num_workers=6)

with torch.no_grad():
        n_correct = 0
        n_total = 0
        for x,y in dataloader_test:
            image = x.to(device)
            label = y.to(device)
            output = model(image)
            pred = 1 if output >= 0.5 else 0  
            n_total += 1
            n_correct += (pred == label).item()
        acc = 100.0 * n_correct / n_total
        print('test accuracy = ',acc)

with torch.no_grad():
    n_correct = 0
    n_total = 0
    for x, y in dataloader_train:
        image = x.to(device)
        label = y.to(device)
        output = model(image)
        pred = 1 if output >= 0.5 else 0
        n_total += 1
        n_correct += (pred == label).item()
    acc = 100.0 * n_correct / n_total
    print('train accuracy = ', acc)

with torch.no_grad():
    n_correct = 0
    n_total = 0
    for x, y in dataloader_val:
        image = x.to(device)
        label = y.to(device)
        output = model(image)
        pred = 1 if output >= 0.5 else 0
        n_total += 1
        n_correct += (pred == label).item()
    acc = 100.0 * n_correct / n_total
    print('val accuracy = ', acc)
