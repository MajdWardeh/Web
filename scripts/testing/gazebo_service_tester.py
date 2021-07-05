import time
from math import pi
import numpy as np
import rospy
from tf.transformations import quaternion_from_euler
from geometry_msgs.msg import PoseStamped, Pose
from std_msgs.msg import Empty
from sensor_msgs.msg import Image
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState

class GazeboServiceTester:
    def __init__(self):
        self.rgbImagesList = []
        self.depthImagesList = []

        rospy.init_node('gazebo_serivce_tester_node', anonymous=True)
        # self.posePub = rospy.Publisher('/uav/command/pose', PoseStamped, queue_size=1)
        # self.armPub = rospy.Publisher('/uav/input/arm', Empty, queue_size=2)
        # self.resetPub = rospy.Publisher('/uav/input/reset', Empty, queue_size=2)
        # self.initalPosePub = rospy.Publisher('/uav/initialPose', Pose, queue_size=2)
        # self.rgbSub = rospy.Subscriber('/uav/camera/left_rgb_blurred/image_rect_color', Image, self.rgbCameraCallback, queue_size=5)
        # self.depthSub = rospy.Subscriber('/uav/camera/left_depth/image_rect_color', Image, self.depthCameraCallback, queue_size=5)
        print('gazebo service tester node started...')

    def rgbCameraCallback(self, msg):
        self.rgbImagesList.append(msg.header.stamp.to_sec())
        self.rgbImagesList = self.rgbImagesList[-10:]

    def depthCameraCallback(self, msg):
        self.depthImagesList.append(msg.header.stamp.to_sec())
        self.depthImagesList = self.depthImagesList[-10:]

    def sendPose(self):
        # rate = rospy.Rate(1)
        # while not rospy.is_shutdown():
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'uav/imu'
        msg.pose.position.x = 1
        msg.pose.position.y = -2
        msg.pose.position.z = 1
        msg.pose.orientation.w = 1
        self.posePub.publish(msg)
        # rate.sleep()

    def sendReset(self):
        self.resetPub.publish(Empty())

    def sendArm(self):
        self.armPub.publish(Empty())
    
    def sendInitialPose(self):
        msg = Pose()
        msg.position.x = 4 
        msg.position.y = -10
        msg.position.z = 1
        # RPY to convert: 90deg, 0, -90deg
        q = quaternion_from_euler(0, 0, 30*pi/180.0)
        print(q)
        msg.orientation.x = q[0] 
        msg.orientation.y = q[1]
        msg.orientation.z = q[2] 
        msg.orientation.w = q[3]
        self.initalPosePub.publish(msg)

    def checkImages(self):
        rate = rospy.Rate(60)
        while not rospy.is_shutdown():
            x = np.array(self.rgbImagesList)
            y = np.array(self.depthImagesList)
            for t1 in x:
                idx = (np.abs(y-t1)).argmin()
                t2 = y[idx]
                print(t1, t2, abs(t1-t2))
            rate.sleep()

    def placeDrone(self, x, y, z):
        state_msg = ModelState()
        state_msg.model_name = 'hummingbird'
        state_msg.pose.position.x = x 
        state_msg.pose.position.y = y
        state_msg.pose.position.z = z
        state_msg.pose.orientation.x = 0
        state_msg.pose.orientation.y = 0
        state_msg.pose.orientation.z = 0
        state_msg.pose.orientation.w = 0
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state(state_msg)
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e



def main():
    gst = GazeboServiceTester()
    gst.placeDrone(0, 0, 10)




if __name__=='__main__':
    main()