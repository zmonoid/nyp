from cv_bridge import CvBridge

import torch
from model import FineTuneModel, IntentionModel
import torchvision.models as models
import torchvision.transforms as transforms
import math
import PIL

from sensor_msgs.msg import Image
import rospy
import std_msgs
import argparse

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('model', metavar='MOD', help='path to model')
args = parser.parse_args()

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transformer = transforms.Compose([
    transforms.Scale(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

model = models.resnet50()
model = IntentionModel(model, 'resnet50', 2)
model = torch.nn.DataParallel(model).cuda()
checkpoint = torch.load(args.model)
model.load_state_dict(checkpoint['state_dict'])
model.eval()
bridge = CvBridge()

out_steer = None
out_vel = None


def listener():
    global out_steer
    global out_vel

    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber('image_raw', Image, callback)
    out_steer = rospy.Publisher('/steer', std_msgs.msg.Float32, queue_size=10)
    out_vel = rospy.Publisher(
        '/desired_velocity_cmd', std_msgs.msg.Float32, queue_size=10)
    rospy.spin()


def callback(data):
    #    try:
    #        image = bridge.imgmsg_to_cv2(data, 'bgr8')
    #    except CvBridgeError as e:

    global out_steer
    global out_vel

    image = bridge.imgmsg_to_cv2(data, 'bgr8')
    image = PIL.Image.fromarray(image)
    intention = torch.FloatTensor([[0, 1]])
    image = transformer(image).view(1, 3, 224, 224)
    input_var = torch.autograd.Variable(image)
    intention_var = torch.autograd.Variable(intention)
    output = model(input_var, intention_var)
    steer = output.data[0].cpu().numpy()[0]
    steer = steer / math.pi * 180.0
    print steer
    out_steer.publish(steer)
    out_vel.publish(0.75)


#    print image

# compute output

if __name__ == '__main__':
    listener()
