from __future__ import print_function
import os
import time, datetime 
from math import pi
import numpy as np
import pandas as pd
import cv2
import rospy
import tf, tf2_ros, tf_conversions
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge
from flightgoggles.msg import IRMarkerArray

def main():
    image = cv2.imread('/home/majd/catkin_ws/src/basic_rl_agent/data/imageMarkersData/ImageMarkersDataset_20210724_123624/images/20210724_121731.jpg')
    k = np.array([548.4088134765625, 0.0, 512.0, 0.0, 548.4088134765625, 384.0, 0.0, 0.0, 1.0]).reshape(3, 3)
    print(k)
    return
    

if __name__ == '__main__':
    main()