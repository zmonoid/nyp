import os
import random
import math
from PIL import Image, ImageEnhance

import torch
import torch.utils.data as data


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageList(data.Dataset):
    def __init__(self,
                 root,
                 flist,
                 transform=None,
                 random_flip=True,
                 random_shift=False,
                 random_brightness=False,
                 random_sharpness=False,
                 target_transform=None,
                 loader=default_loader,
                 imsize=(224, 224)):

        imgs = []
        list_path = os.path.join(root, flist)
        with open(list_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                path = os.path.join(root, line.split(' ')[0].strip())
                angle = float(line.split(' ')[-1].strip())
                angle = angle / 180.0 * math.pi
                target = torch.torch.FloatTensor([angle])
                item = (path, target)
                imgs.append(item)

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.random_flip = random_flip
        self.random_brightness = random_brightness
        self.random_sharpness = random_sharpness
        self.random_shift = random_shift
        self.target_transform = target_transform
        self.loader = loader
        self.imsize = imsize

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)

        # Random Flip

        if random.random() > 0.5 and self.random_flip:
            target = -target
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # Random Shift

        if self.random_shift:
            factor = random.betavariate(2, 2)
            w, h = img.size
            diff = w - h
            x1 = int(factor * diff)
            img = img.crop((x1, 0, x1 + h, h))
            target -= (factor * 2 - 1) * math.pi

        # Random Brightness

        if self.random_brightness:
            factor = random.betavariate(2, 2)
            bright = ImageEnhance.Brightness(img)
            img = bright.enhance(factor)

        # Random Sharpness

        if self.random_sharpness:
            factor = random.betavariate(2, 2)
            sharp = ImageEnhance.Sharpness(img)
            img = sharp.enhance(factor * 2)

        img = img.resize(self.imsize)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        #import pdb
        #pdb.set_trace()
        return img, target

    def __len__(self):
        return len(self.imgs)
