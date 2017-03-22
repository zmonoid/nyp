# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import torchvision.models as models
import torchvision.transforms as transforms
import torch
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from model import HeatmapModel, FineTuneModel
import argparse

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('mod', metavar='MOD', help='checkpoint name')
parser.add_argument('save', metavar='SAV', help='save directory')

args = parser.parse_args()

model = models.resnet50()
model = FineTuneModel(model, 'resnet50', 1)
model = torch.nn.DataParallel(model).cuda()
checkpoint = torch.load(args.mod)
model.load_state_dict(checkpoint['state_dict'])
model = HeatmapModel(model)
model = torch.nn.DataParallel(model).cuda()
model.eval()

weight = checkpoint['state_dict']['module.classifier.1.weight'].cpu().numpy()[
    0]
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transformer = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

f = open(os.path.join(args.data, 'data.csv'), 'r')
lines = f.readlines()

for idx, line in enumerate(lines):
    img = os.path.join(args.data, line.split(' ')[0].strip())
    image = Image.open(img)
    image = transformer(image).view(1, 3, 640, 480)
    input_var = torch.autograd.Variable(image)
    output = model(input_var)
    feat = output.data[0].cpu().numpy()
    heat = sum(map(lambda x, y: x * y, feat, weight))
    im = cv2.imread(img)
    rgb = cv2.cvtColor(cv2.resize(im, (640, 480)), cv2.COLOR_BGR2RGB)
    heat = cv2.resize(heat, (640, 480))
    max_response = heat.mean()
    heat /= heat.max()
    im_show = rgb.astype(np.float32) / 255 * 0.3 + plt.cm.jet(
        heat / heat.max())[:, :, :3] * 0.7

    plt.imsave(
        os.path.join(args.save, 'haha_%05d.jpg' % idx), im_show, format='jpg')
    print idx, len(lines)
