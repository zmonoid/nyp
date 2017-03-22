#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2016 Massachusetts Institute of Technology
"""Extract images from a rosbag.
"""

import os
import argparse

import cv2

import rosbag
from cv_bridge import CvBridge


def main():
    """Extract a folder of images from a rosbag.
    """
    parser = argparse.ArgumentParser(
        description="Extract images from a ROS bag.")
    parser.add_argument("bags_dir", help="input directory of ros bags.")
    parser.add_argument("output_dir", help="output directory of images.")
    parser.add_argument(
        "--image_topic", help="Image topic.", default='/image_raw')
    parser.add_argument(
        "--steer_topic", help="Steer topic.", default='/actual_str')

    args = parser.parse_args()

    filename = os.path.join(args.output_dir, 'data.csv')
    f = open(filename, 'w')

    count = 0
    n = 0

    for bagfile in os.listdir(args.bags_dir):

        bagfile = os.path.join(args.bags_dir, bagfile)
        print "Extract images from %s on topic %s into %s" % (
            bagfile, args.image_topic, args.output_dir)
        bag = rosbag.Bag(bagfile, "r")
        bridge = CvBridge()
        last_steer = 0
        for topic, msg, t in bag.read_messages(
                topics=[args.image_topic, args.steer_topic]):

            if topic == args.image_topic:
                cv_img = bridge.imgmsg_to_cv2(msg, "bgr8")
                image_name = "%s.jpg" % str(t)
                cv2.imwrite(os.path.join(args.output_dir, image_name), cv_img)
                f.write("%s %f %d\n" % (image_name, last_steer, n))
                print "Wrote image %i" % count
                count += 1

            elif topic == args.steer_topic:
                last_steer = msg.data

        bag.close()
        n += 1

    return


if __name__ == '__main__':
    main()
