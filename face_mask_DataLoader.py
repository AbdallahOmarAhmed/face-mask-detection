import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from torchvision.io import read_image
from torchvision import transforms

class faceMaskDataSet(Dataset):
    pathMask = 'Face Mask Dataset/Train/WithMask/'
    pathNoMask = 'Face Mask Dataset/Train/WithoutMask/'
    def __init__(self):
        self.myTransforms =  transforms.Resize(256)
        self.maskData = os.listdir(faceMaskDataSet.pathMask)
        self.noMaskData = os.listdir(faceMaskDataSet.pathNoMask)
        self.maskLabel = np.ones(len(self.maskData))
        self.noMaskLabel = np.zeros(len(self.noMaskData))
        self.X = self.maskData + self.noMaskData
        self.Y = np.concatenate((self.maskLabel, self.noMaskLabel))
    def __len__(self):
        return len(self.X)
    def __getitem__(self,index): 
        self.y = self.Y[index]
        self.filePath = "Face Mask Dataset/Train/WithMask/" if self.y==1.\
            else "Face Mask Dataset/Train/WithoutMask/"
        self.path = self.X[index]
        self.image = read_image(self.filePath + self.path)
        self.x = self.myTransforms(self.image)/255
        return self.x, self.y


class faceMaskTestSet(Dataset):
    pathMask = 'Face Mask Dataset/Test/WithMask/'
    pathNoMask = 'Face Mask Dataset/Test/WithoutMask/'
    def __init__(self):
        self.myTransforms =  transforms.Resize(256)
        self.maskData = os.listdir(faceMaskTestSet.pathMask)
        self.noMaskData = os.listdir(faceMaskTestSet.pathNoMask)
        self.maskLabel = np.ones(len(self.maskData))
        self.noMaskLabel = np.zeros(len(self.noMaskData))
        self.X = self.maskData + self.noMaskData
        self.Y = np.concatenate((self.maskLabel, self.noMaskLabel))
    def __len__(self):
        return len(self.X)
    def __getitem__(self,index):     
        self.y = self.Y[index]
        self.filePath = "Face Mask Dataset/Test/WithMask/" if self.y==1\
            else "Face Mask Dataset/Test/WithoutMask/"
        self.path = self.X[index]
        self.image = read_image(self.filePath + self.path)
        self.x = self.myTransforms(self.image)/255
        return self.x, self.y


class faceMaskAccSet(Dataset):
    pathMask = 'Face Mask Dataset/Validation/WithMask/'
    pathNoMask = 'Face Mask Dataset/Validation/WithoutMask/'
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
        self.filePath = "Face Mask Dataset/Validation/WithMask/" if self.y==1\
            else "Face Mask Dataset/Validation/WithoutMask/"
        self.path = self.X[index]
        self.image = read_image(self.filePath + self.path)
        self.x = self.myTransforms(self.image)/255
        return self.x, self.y