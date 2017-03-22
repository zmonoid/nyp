import torch
import torchvision.transforms as transforms
import torchvision.models as models
from model import FineTuneModel
from loader import ImageList
import numpy as np

import argparse
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('model', metavar='MOD', help='name of model')
parser.add_argument('dir', metavar='DIR', help='directory of dataset')
args = parser.parse_args()

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
val_loader = torch.utils.data.DataLoader(
    ImageList(
        args.dir,
        'data.csv',
        transforms.Compose([
            #transforms.Scale(224),
            #transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]),
        random_flip=False),
    batch_size=512,
    shuffle=False,
    num_workers=4,
    pin_memory=True)

model = models.resnet50()
model = FineTuneModel(model, 'resnet50', 1)
model = torch.nn.DataParallel(model).cuda()
checkpoint = torch.load(args.model)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

targets = []
outputs = []

for i, (input, target) in enumerate(val_loader):
    target = target.cuda(async=True)
    input_var = torch.autograd.Variable(input, volatile=True)
    target_var = torch.autograd.Variable(target, volatile=True)

    # compute output
    output = model(input_var)

    for out_ in output.data.tolist():
        outputs.append(out_[0])
    for target_ in target.tolist():
        targets.append(target_[0])

    print i, len(val_loader)

targets = np.asarray(targets)
outputs = np.asarray(outputs)
np.save('steer.npy', (targets, outputs))
