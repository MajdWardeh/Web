import time
from math import pi
import numpy as np
import rospy
from tf.transformations import quaternion_from_euler
from geometry_msgs.msg import PoseStamped, Pose
from std_msgs.msg import Empty
from sensor_msgs.msg import Image
import tf_conversions
import tf2_ros
import geometry_msgs.msg

class TF_Sender:
    def __init__(self):
        rospy.init_node('tf_sender_node', anonymous=True)
        self.poseSubs = rospy.Subscriber('/uav/pose', Pose, self.poseCallback, queue_size=1)

        self.poseUpdated = False

        print('tf sender node started...')
        time.sleep(0.5)

    def poseCallback(self, msg):
        x = msg.position.x
        y = msg.position.y
        z = msg.position.z
        qx = msg.orientation.x
        qy = msg.orientation.y
        qz = msg.orientation.z
        qw = msg.orientation.w
        q = [qx, qy, qz, qw]
        print('new pose recieved', x, y, z, q)
        self.x, self.y, self.z = x, y, z
        self.q = q
        self.poseUpdated = True



    def sendPose(self):
        br = tf2_ros.TransformBroadcaster()
        t = geometry_msgs.msg.TransformStamped()

        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "world"
        t.child_frame_id = "uav/imu"
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = self.z

        if self.q is None: 
            print("q is none")
            roll, pitch, yaw = 0, 0, self.yaw
            roll, pitch, yaw = [angle*pi/180.0 for angle in [roll, pitch, yaw]]
            self.q = tf_conversions.transformations.quaternion_from_euler(roll, pitch, yaw)
        t.transform.rotation.x = self.q[0]
        t.transform.rotation.y = self.q[1]
        t.transform.rotation.z = self.q[2]
        t.transform.rotation.w = self.q[3]

        br.sendTransform(t)
    
    def generateRandomPose(self, gateX, gateY, gateZ):
        return (-5.422825041479353, 4.632026502854284, 1.67661692851683, 74.71197863869756)
        xmin, xmax = -10, 10
        ymin, ymax = -10, 10
        zmin, zmax = 1, 2 
        x = xmin + np.random.rand() * (xmax - xmin)
        y = ymin + np.random.rand() * (ymax - ymin)
        z = zmin + np.random.rand() * (zmax - zmin)
        # z_segma = 2/5
        # z = np.random.normal(gateZ, z_segma) 
        yaw_min, yaw_max = -180, 180
        yaw = yaw_min + np.random.rand() * (yaw_max - yaw_min)
        return x, y, z, yaw

    def updatePose(self, x, y, z, yaw):
        self.x = x
        self.y = y
        self.z = z
        self.yaw = yaw
        self.q = None

def getQ(yaw):
    yaw = yaw * pi/180.0
    q = tf_conversions.transformations.quaternion_from_euler(0, 0, yaw)
    return q[0], q[1], q[2], q[3]

def main():
    tfSender = TF_Sender()
    r = rospy.Rate(1000)
    x, y, z, yaw = tfSender.generateRandomPose(gateX=30.7, gateY=10, gateZ=2.4)
    tfSender.updatePose(x, y, z, yaw)
    while not rospy.is_shutdown():
        tfSender.sendPose()
        r.sleep()




if __name__=='__main__':
    main()