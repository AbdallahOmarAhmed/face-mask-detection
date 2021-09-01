import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import os
from torchvision.io import read_image
from torchvision import transforms

class faceMaskDataSet(Dataset):
    pathMask = 'faceMaskDataSet/Train/WithMask/'
    pathNoMask = 'faceMaskDataSet/Train/WithoutMask/'
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
        self.filePath = "faceMaskDataSet/Train/WithMask/" if self.y==1.\
            else "faceMaskDataSet/Train/WithoutMask/"       
        self.path = self.X[index]
        self.image = read_image(self.filePath + self.path)
        self.x = self.myTransforms(self.image)/255
        return self.x, self.y


class faceMaskTestSet(Dataset):
    pathMask = 'faceMaskDataSet/Test/WithMask/'
    pathNoMask = 'faceMaskDataSet/Test/WithoutMask/'
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
        self.filePath = "faceMaskDataSet/Test/WithMask/" if self.y==1\
            else "faceMaskDataSet/Test/WithoutMask/"       
        self.path = self.X[index]
        self.image = read_image(self.filePath + self.path)
        self.x = self.myTransforms(self.image)/255
        return self.x, self.y


class faceMaskAccSet(Dataset):
    pathMask = 'faceMaskDataSet/Validation/WithMask/'
    pathNoMask = 'faceMaskDataSet/Validation/WithoutMask/'
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
        self.filePath = "faceMaskDataSet/Validation/WithMask/" if self.y==1\
            else "faceMaskDataSet/Validation/WithoutMask/"       
        self.path = self.X[index]
        self.image = read_image(self.filePath + self.path)
        self.x = self.myTransforms(self.image)/255
        return self.x, self.y