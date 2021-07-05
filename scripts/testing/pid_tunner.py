from math import pi as PI
import time
import rospy
import roslaunch
import dynamic_reconfigure.client
import tf
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty
from mav_msgs.msg import RollPitchYawrateThrust

class PID_tunner:
    def __init__(self):
        rospy.init_node('PID_tunner_node', anonymous=True)

        self.rollRef = 0
        self.pitchRef = 0
        self.rollErrorAbsSum = 0
        self.pitchErrorAbsSum = 0

        self.odomSub = rospy.Subscriber('/uav/odometry', Odometry, self.odometryCallback, queue_size=1)
        # self.attitudeThrustReferenceSub = rospy.Subscriber('/uav/command/roll_pitch_yawrate_thrust', RollPitchYawrateThrust, self.attitudeRefCallback, queue_size=1)
        self.armPub = rospy.Publisher('/uav/input/arm', Empty, queue_size=2)
        self.resetPub = rospy.Publisher('/uav/input/reset', Empty, queue_size=2)
        # self.posePub = rospy.Publisher('/uav/command/pose', PoseStamped, queue_size=1)
        self.rpyRateThrustPub = rospy.Publisher('/uav/command/roll_pitch_yawrate_thrust', RollPitchYawrateThrust, queue_size=1)

        print('PID tunner node started...')
        time.sleep(0.5)

    def odometryCallback(self, msg):
        pose = msg.pose.pose
        quaternion = (
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        rollError = self.rollRef - euler[0]
        pitchError = self.pitchRef - euler[1]

        self.rollErrorAbsSum += abs(rollError)
        self.pitchErrorAbsSum += abs(pitchError)
        # print(rollError, pitchError, self.rollErrorAbsSum, self.pitchErrorAbsSum)

    def attitudeRefCallback(self, msg):
        self.rollRef = msg.roll
        self.pitchRef = msg.pitch

    def sendPose(self):
        # rate = rospy.Rate(1)
        # while not rospy.is_shutdown():
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'uav/imu'
        msg.pose.position.x = 0
        msg.pose.position.y = 0
        msg.pose.position.z = 10
        self.posePub.publish(msg)
        # rate.sleep()
        

    def sendRPYrateThrust(self, roll=0, pitch=0, yawRate=0, thrust=9.9):
        msg = RollPitchYawrateThrust()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'uav/imu'
        msg.roll = roll 
        msg.pitch = pitch
        msg.yaw_rate = yawRate
        msg.thrust.z = thrust
        self.rpyRateThrustPub.publish(msg)
        # variables updating
        self.rollRef = roll
        self.pitchRef = pitch
        self.rollErrorAbsSum = 0
        self.pitchErrorAbsSum = 0

    def startPIDController(self):
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        launch = roslaunch.parent.ROSLaunchParent(uuid, ["/home/majd/catkin_ws/src/flightgoggles/flightgoggles/launch/PID_attitude_controller.launch"], verbose=True)
        return launch


    def tune(self):
        for rollGain in [40, 30, 20, 10]:
            self.resetPub.publish(Empty())
            launch = self.startPIDController()
            launch.start()
            time.sleep(1)
            self.client = dynamic_reconfigure.client.Client("/uav/PID_attitude_controller", timeout=30, config_callback=self.dynamicReconfigureCallback)
            self.client.update_configuration({"roll_gain":(rollGain)})
            self.sendRPYrateThrust(roll=20*PI/180.0)
            self.rollErrorAbsSum = 0
            time.sleep(0.5)
            self.armPub.publish(Empty())
            rospy.sleep(50)
            launch.shutdown()
            self.client.close()
            rospy.logwarn('rollGain={}, rollErrorAbsSum={}'.format(rollGain, self.rollErrorAbsSum) )
            time.sleep(1)
        
            
    def dynamicReconfigureCallback(self, config):
        rospy.loginfo("Config set to {roll_gain}".format(**config))
    
    def spin(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            rate.sleep() 
    
        

def main():
    pidTunner = PID_tunner()
    pidTunner.tune()
    # pidTunner.spin()



if __name__=='__main__':
	main()