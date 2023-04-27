import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image

from face_mask_DataLoader import faceMaskTestSet

import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models






model = models.resnet18()
n = model.fc.in_features
model.fc = nn.Linear(n, 1)
model.load_state_dict(torch.load("weights/modelTR.pth"))
model.eval()
# model = model.to(memory_format=torch.channels_last)

class faceMaskAccSet(Dataset):
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


data = faceMaskAccSet()
dataLoader = DataLoader(data, batch_size=1, num_workers=8)

datatest = faceMaskTestSet()
testLoader = DataLoader(datatest, batch_size=1, num_workers=8)
# train_sampler = torch.utils.data.distributed.DistributedSampler(data)


train_loader = torch.utils.data.DataLoader(
    data, batch_size=1, shuffle=True,
     pin_memory=True)

def eval_func(model):
    with torch.no_grad():
        n_correct = 0
        n_total = 0
        for x, y in testLoader:
            output = model(x)
            pred = 1 if output >= 0.5 else 0
            n_total += 1
            n_correct += (pred == y).item()
        acc = 100.0 * n_correct / n_total
        return acc

from neural_compressor import PostTrainingQuantConfig
from neural_compressor import quantization
conf = PostTrainingQuantConfig(backend='ipex')
# import ipdb;ipdb.set_trace()

q_model = quantization.fit(model,
                           conf,
                           calib_dataloader=dataLoader,
                           eval_func=eval_func)
q_model.save('test2')
