import os

import numpy as np
import timm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.io import read_image
from torchvision.transforms import transforms

from face_mask_DataLoader import faceMaskTestSet


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

model = models.resnet18()
n = model.fc.in_features
model.fc = nn.Linear(n, 1)
model.load_state_dict(torch.load("weights/modelTR.pth"))
model.eval()
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
conf = PostTrainingQuantConfig()
q_model = quantization.fit(model,
                           conf,
                           calib_dataloader=dataLoader)
q_model.save("test2")


#   warnings.warn(
# /home/abdallah/.local/lib/python3.8/site-packages/intel_extension_for_pytorch/quantization/_quantization_state_utils.py:362:
# TracerWarning: Converting a tensor to a Python number might cause the trace to be incorrect.
# We can't record the data flow of Python values, so this value will be treated as a constant in the future.
# This means that the trace might not generalize to other inputs!
#   args = torch.quantize_per_tensor(args, scale.item(), zp.item(), dtype)



# /home/abdallah/.local/lib/python3.8/site-packages/intel_extension_for_pytorch/quantization/_quantization_state.py:360:
# TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect.
# We can't record the data flow of Python values, so this value will be treated as a constant in the future.
# This means that the trace might not generalize to other inputs!
#   if scale.numel() > 1:

# /home/abdallah/.local/lib/python3.8/site-packages/intel_extension_for_pytorch/quantization/_quantization_state_utils.py:370:
# TracerWarning: Converting a tensor to a Python number might cause the trace to be incorrect.
# We can't record the data flow of Python values, so this value will be treated as a constant in the future.
# This means that the trace might not generalize to other inputs!
#   args = torch.quantize_per_tensor(args, scale.item(), zp.item(), dtype)

# /home/abdallah/.local/lib/python3.8/site-packages/intel_extension_for_pytorch/quantization/_quantization_state.py:455: TracerWarning: Converting a tensor to a Python number might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
#   output, scale.item(), zp.item(), inf_dtype)
# /home/abdallah/.local/lib/python3.8/site-packages/intel_extension_for_pytorch/quantization/_quantization_state.py:363: TracerWarning: Converting a tensor to a Python number might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
#   arg = torch.quantize_per_tensor(weight, scale.item(), zp.item(), dtype)
