import time
import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Empty

class GatePoseSender:
    def __init__(self):
        rospy.init_node('gate_pose_sender_node', anonymous=True)
        self.gatePosePub = rospy.Publisher('/gate_pose', PoseStamped, queue_size=1)
        print('gate pose sender node started...')
        time.sleep(0.5)

    def sendPose(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            msg = PoseStamped()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = 'uav/imu'
            msg.pose.position.x = 6
            msg.pose.position.y = 0
            msg.pose.position.z = 10
            self.posePub.publish(msg)
        rate.sleep()

    def sendReset(self):
        self.resetPub.publish(Empty())

    def sendArm(self):
        self.armPub.publish(Empty())
    


def main():
    poseSender = PoseSender()
    poseSender.sendReset()
    time.sleep(0.5)
    poseSender.sendPose()
    time.sleep(0.5)
    poseSender.sendArm()



if __name__=='__main__':
	main()