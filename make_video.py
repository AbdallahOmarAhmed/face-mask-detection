import torch
import numpy as np
import torchvision
import os
import torch.nn as nn
from torchvision.io import read_image
import face_recognition
from PIL import Image
import cv2
from torchvision import transforms
from face_box import load_model
from face_box import*
from models.faceboxes import FaceBoxes
from utils.box_utils import decode
from layers.functions.prior_box import PriorBox
from data import cfg
from utils.nms_wrapper import nms


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

torch.set_grad_enabled(False)
totensor = transforms.ToTensor()
myTransforms = transforms.Resize(256)
model = torchvision.models.resnet18()
n = model.fc.in_features
model.fc = nn.Linear(n, 1)
model.load_state_dict(torch.load("weights/modelTR.pth"))
model = model.to(device)
model.eval()

net = FaceBoxes(phase='test', size=None, num_classes=2)  # initialize detector
net = load_model(net, 'weights/FaceBoxes.pth', False)
net.eval()
net = net.to(device)
print('Finished loading model!')



def crop(image):#cv np array
    img = np.float32(image)
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)
    loc, conf = net(img)
    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

    order = scores.argsort()[::-1][:2]
    boxes = boxes[order]
    scores = scores[order]

    inds = np.where(scores > 0.05)[0]
    boxes = boxes[inds]
    scores = scores[inds]

    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = nms(dets, 0.3, force_cpu=False)
    dets = dets[keep, :]
    dets = dets[:750:]

    # top, right, bottom, left = face_locations[i]
    # x, y, width, height = face_locations[i]
    # x2, y2 = x + width, y + height
    # face_image = image[y:y2, x:x2]
    # imgs.append(face_image)
    return dets

def runModel(dets,frame):
    out = []
    width = 256
    height = 256
    dim = (width, height)
    for k in dets:
        if k[4] < 0.6:
            continue
        k = list(map(int, k))
        face_image = frame[k[1]:k[3], k[0]:k[2]]
        face_image = cv2.resize(face_image, dim, interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        #img2 = myTransforms(img)
        #img.show()
        img = totensor(img)
        img = img.unsqueeze(0)
        img = img.to(device)
        output = model(img)
        out.append(output)

        # img = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
        # img = Image.fromarray(img)
        # img = myTransforms(img)
        # img = totensor(img)
        # img = img.unsqueeze(0)
        # img = img.to(device)
        # output = model(img)
        # out.append(output)
    return out

def draw(frame, output, face_locations):
    for i in range(len(output)):
        k = face_locations[i]
        if k[4] < 0.6:
            continue
        k = list(map(int, k))
        color = (0,255,0) if output[i] >= 0.5 else (0,0,255)
        cv2.rectangle(frame, (k[0], k[1]), (k[2], k[3]), color, 4)
        #        cv2.rectangle(frame, (left, top), (right, bottom), color, 4)
    return frame

vidcap = cv2.VideoCapture('test.mp4')
success,frame = vidcap.read()
video = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'),\
 30,(frame.shape[1],frame.shape[0]))
count = 0
while success:
    success,frame = vidcap.read()
    if type(frame) == type(None):
        break
    count += 1
    print("frame :",count)
    face_locations = crop(frame)
    output = runModel(face_locations,frame)
    frame = draw(frame, output, face_locations)
    video.write(frame)
print('done !')
