import time
from math import pi
import numpy as np
import rospy
from tf.transformations import quaternion_from_euler
from geometry_msgs.msg import PoseStamped, Pose
from std_msgs.msg import Empty
from sensor_msgs.msg import Image

class PoseSender:
    def __init__(self):
        self.rgbImagesList = []
        self.depthImagesList = []

        rospy.init_node('pose_sender_node', anonymous=True)
        self.posePub = rospy.Publisher('/uav/command/pose', PoseStamped, queue_size=1)
        self.armPub = rospy.Publisher('/uav/input/arm', Empty, queue_size=2)
        self.resetPub = rospy.Publisher('/uav/input/reset', Empty, queue_size=2)
        self.initalPosePub = rospy.Publisher('/uav/initialPose', Pose, queue_size=2)
        self.rgbSub = rospy.Subscriber('/uav/camera/left_rgb_blurred/image_rect_color', Image, self.rgbCameraCallback, queue_size=5)
        self.depthSub = rospy.Subscriber('/uav/camera/left_depth/image_rect_color', Image, self.depthCameraCallback, queue_size=5)
        print('pose sender node started...')
        time.sleep(0.5)

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
    
    def sendInitialPose(self, x, y, z, yaw):
        msg = Pose()
        msg.position.x = x
        msg.position.y = y
        msg.position.z = z 
        # RPY to convert: 90deg, 0, -90deg
        q = quaternion_from_euler(0, 0, yaw*pi/180.0)
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




def main():
    posesender = PoseSender()
    # posesender.checkimages()
    # posesender.sendreset()
    # time.sleep(0.5)
    # posesender.sendpose()
    # time.sleep(0.5)
    # posesender.sendarm()
    
    posesender.sendInitialPose(x=10, y=10, z=2.4, yaw=0)



if __name__=='__main__':
    main()