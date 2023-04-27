from timeit import default_timer as timer

start = timer()
import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import os
from torchvision.io import read_image
from torchvision.transforms import transforms
from tqdm import tqdm

from face_mask_DataLoader import faceMaskAccSet, faceMaskTestSet, faceMaskDataSet
from neural_compressor.utils.pytorch import load


class faceMaskAccSet2(Dataset):
    pathMask = 'archive/Face Mask Dataset/Validation/WithMask/'
    pathNoMask = 'archive/Face Mask Dataset/Validation/WithoutMask/'
    def __init__(self):
        self.myTransforms =  transforms.Resize(256)
        self.maskData = os.listdir(faceMaskAccSet.pathMask)
        self.noMaskData = os.listdir(faceMaskAccSet.pathNoMask)
        self.maskLabel = np.ones(len(self.maskData))
        self.noMaskLabel = np.zeros(len(self.noMaskData))
        self.X = self.maskData + self.noMaskData
        self.Y = np.concatenate((self.maskLabel, self.noMaskLabel))
    def __len__(self):
        return len(self.X)
    def __getitem__(self,index):
        self.y = self.Y[index]
        self.filePath = "archive/Face Mask Dataset/Validation/WithMask/" if self.y==1\
            else "archive/Face Mask Dataset/Validation/WithoutMask/"
        self.path = self.X[index]
        self.image = read_image(self.filePath + self.path)
        self.x = self.myTransforms(self.image)/255
        return self.x

device = torch.device('cpu') # 'cuda' if torch.cuda.is_available() else
print(device)
data = faceMaskAccSet2()
dataLoader = DataLoader(data, batch_size=100, num_workers=8)
#load the model
model = torchvision.models.resnet18()
n = model.fc.in_features
model.fc = nn.Linear(n, 1)
model.load_state_dict(torch.load("weights/modelTR.pth"))
# q_model = load('test',
#                         model,
#                         dataloader=dataLoader)
# q_model.eval()
# q_model.to(device)
model.eval()

x = torch.randn(1,3,1024,1024)
start = timer()
model(x)
end = timer()
print(end-start)

#
# data_train = faceMaskDataSet()
# dataloader_train = DataLoader(data_train, batch_size=100,\
#  shuffle=False, num_workers=6)
#
# start = timer()
# with torch.no_grad():
#     n_correct = 0
#     n_total = 0
#     for x, y in tqdm(dataloader_train):
#         output = model(x)
#         pred = output >= 0.5
#         n_total += 1
#         n_correct += torch.sum((pred == y)).item()
#
#     acc = 100.0 * n_correct / n_total
#     print('train accuracy = ', acc)
# end = timer()
# print(end-start)