import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import os
from torchvision.io import read_image
from torchvision import transforms, models
from face_mask_DataLoader import faceMaskDataSet, faceMaskTestSet

def main():
    pass

if __name__ == "__main__":
    # 'cuda' if torch.cuda.is_available() else
    device = torch.device('cuda' if torch.cuda.is_available() else'cpu')
    print(device)

    # hayper parametars
    num_epochs = 100
    learning_rate = 0.001
    batch_size = 50

    # load data
    train_data = faceMaskDataSet()
    test_data = faceMaskTestSet()

    train_dataloader = DataLoader(train_data, batch_size=batch_size,shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_data, batch_size=1,shuffle=False, num_workers=0)
    main()


# create model


if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    # init model
    model = models.resnet18()
    n = model.fc.in_features
    model.fc = nn.Linear(n, 1)
    Loss = nn.BCELoss()
    optimaizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimaizer, step_size=10, gamma=0.5)
    model.to(device)
    max = 90.

    # train loop
    for epoch in range(num_epochs):
        model.train()
        for x,y in train_dataloader:
            # forward
            images = x.to(torch.float32).to(device)
            labels = y.reshape(-1,1).to(torch.float32).to(device)
            output = model(images)
            output = torch.sigmoid(output)
            loss = Loss(output, labels)
            # backward
            optimaizer.zero_grad()
            loss.backward()
            optimaizer.step()
        scheduler.step()
        print(epoch + 1, '/', num_epochs, 'loss = ', loss.item())
        model.eval()
        # accuracy
        with torch.no_grad():
            n_correct = 0
            n_total = 0
            for x,y in test_dataloader:
                image = x.to(device)
                label = y.to(device)
                output = model(image)
                pred = 1 if output >= 0.5 else 0
                n_total += 1
                n_correct += (pred == label).item()
            acc = 100.0 * n_correct / n_total
            print('accuracy = ',acc)
            if acc > max :
                max = acc
                torch.save(model.state_dict(),'weights/modelTR.pth')
    main()
