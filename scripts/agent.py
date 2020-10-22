#!/usr/bin/env python

import rospy

from geometry_msgs.msg import TwistStamped
from geometry_msgs.msg import Vector3
from std_msgs.msg import Empty, Bool
from sensor_msgs.msg import Image

last_msg_stamp = 10
def CameraCallback(msg):
    global last_msg_stamp
    current_stamp = msg.header.stamp.to_nsec()
    time_diff = current_stamp - last_msg_stamp 
    last_msg_stamp = current_stamp
    #print(1.0/time_diff)


def main():
    pub_velocity = rospy.Publisher('/hummingbird/autopilot/velocity_command', TwistStamped, queue_size=10)
    pub_land = rospy.Publisher('/hummingbird/autopilot/land', Empty, queue_size=1)
    pub_start = rospy.Publisher('/hummingbird/autopilot/start', Empty, queue_size=1)
    pub_arm_bridge = rospy.Publisher('/hummingbird/bridge/arm', Bool, queue_size=1)
    rospy.Subscriber('/hummingbird/rgb_camera/camera_1/image_raw', Image, CameraCallback) 
    rospy.init_node('RL_Ageint_controller', anonymous=True)
    rate = rospy.Rate(10) # 10hz
     
    pub_arm_bridge.publish(Bool(True))
    rospy.sleep(0.5)
    pub_start.publish(Empty())
    rospy.sleep(0.5)
    while not rospy.is_shutdown():
        msg = TwistStamped()
	msg.header.stamp = rospy.Time.now()
	msg.twist.linear.x = -0.8
	msg.twist.angular.z = 1.0
        pub_velocity.publish(msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
