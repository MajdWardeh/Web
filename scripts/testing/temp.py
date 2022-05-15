import os
import signal
import sys
import math
import numpy as np
from numpy import linalg as la
import time
import datetime
import subprocess
import shutil
from scipy.spatial.transform import Rotation
import math
from math import floor
import rospy
import roslaunch
import tf
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, MultiArrayLayout
from geometry_msgs.msg import PoseStamped, Pose, Quaternion
from gazebo_msgs.msg import ModelState, LinkStates
from mav_planning_msgs.msg import PolynomialTrajectory4D
from nav_msgs.msg import Path, Odometry
from trajectory_msgs.msg import MultiDOFJointTrajectory
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelState




class WorldDataCollector:

    def __init__(self):
        rospy.init_node('data_collector', anonymous=True)

        self.gate_poses_updated = False
        
        self.linkStatesSub = rospy.Subscriber('/gazebo/link_states', LinkStates, self.linkStatesCallback)

    def linkStatesCallback(self, msg):
        self.gate_names = {}
        for i, name in enumerate(msg.name):
            if 'gate' in name:
                self.gate_names[i] = name

        gate_indices = self.gate_names.keys()
        self.gate_positions = {}
        self.gate_headings = {}
        for i, pose in enumerate(msg.pose):
            if i in gate_indices:
                self.gate_positions[i] = (pose.position.x, pose.position.y, pose.position.z)
                rot = Rotation.from_quat([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
                yaw = rot.as_euler('xyz')[-1]
                self.gate_headings[i] = yaw
        self.gate_poses_updated = True
        self.linkStatesSub.unregister()

    def generate_file(self):
        rate = rospy.Rate(1)
        while not self.gate_poses_updated:
            rate.sleep()

        gate_indices = self.gate_names.keys()
        gate_indices.sort()

        with open('goals1.yaml', 'w') as f:
            f.write('goal_positions:\n')
            for i in gate_indices:
                p = self.gate_positions[i]
                f.write('  - {{x: {}, y: {}, z: {}, gate: 1.0}}\n'.format(p[0], p[1], p[2]))
            f.write('\n')
            
            f.write('goal_orientations:\n')
            for i in gate_indices:
                yaw = self.gate_headings[i]
                f.write('  - {{yaw: {}, offset: 0.0}}\n'.format(yaw))
        
def main():
    wdc = WorldDataCollector()
    wdc.generate_file()


if __name__ == '__main__':
    main()