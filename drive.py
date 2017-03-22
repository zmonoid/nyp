import argparse
import base64
from datetime import datetime
import os
import shutil
import numpy as np

import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None


class FineTuneModel(nn.Module):
    def __init__(self, original_model, arch, num_classes):
        super(FineTuneModel, self).__init__()

        if arch.startswith('alexnet'):
            self.features = original_model.features
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes), )
            self.modelName = 'alexnet'
        elif arch.startswith('resnet'):
            # Everything except the last linear layer
            self.features = nn.Sequential(*list(original_model.children())[:
                                                                           -1])
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(512 * 4, 100),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(100, num_classes), )

            self.modelName = 'resnet'
        else:
            raise ("Finetuning not supported on this architecture yet")

        # Freeze those weights
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x):
        f = self.features(x)
        if self.modelName == 'alexnet':
            f = f.view(f.size(0), 256 * 6 * 6)
        elif self.modelName == 'resnet':
            f = f.view(f.size(0), -1)
        y = self.classifier(f)
        return torch.atan(y) * 2


model = models.resnet50()
model = FineTuneModel(model, 'resnet50', 1)
model = torch.nn.DataParallel(model).cuda()
checkpoint = torch.load('../data/model_best.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
model.eval()

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transformer = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])


class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement

        # integral error
        self.integral += self.error

        return self.Kp * self.error + self.Ki * self.integral


controller = SimplePIController(0.1, 0.002)
set_speed = 15
controller.set_desired(set_speed)

steers = []


@sio.on('telemetry')
def telemetry(sid, data):
    global steer

    if data:
        # The current steering angle of the car
        # steering_angle = data["steering_angle"]
        # The current throttle of the car
        throttle = data["throttle"]
        # The current speed of the car
        speed = data["speed"]
        # The current image from the center camera of the car
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        img = transformer(image).view(1, 3, 224, 224)
        input = torch.autograd.Variable(img)
        steer = model(input)
        steer = float(steer.data[0].cpu().numpy())
        steers.append(steer)

        if len(steers) > 5:
            steers.pop()

        throttle = controller.update(float(speed))

        print(steer, throttle)
        send_control(steer, throttle)

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
