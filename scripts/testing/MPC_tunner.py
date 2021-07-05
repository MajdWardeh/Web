import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
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

class MPC_tunner:
    def __init__(self):
        rospy.init_node('MPC_tunner_node', anonymous=True)

        self.rollRef = 0
        self.pitchRef = 0
        self.rollErrorAbsSum = 0
        self.pitchErrorAbsSum = 0
        self.timeRollDect = {}
        self.start_saving_data = False
        self.stop_saving_data = False
        self.counter = 0


        self.odomSub = rospy.Subscriber('/uav/odometry', Odometry, self.odometryCallback, queue_size=20)
        # self.attitudeThrustReferenceSub = rospy.Subscriber('/uav/command/roll_pitch_yawrate_thrust', RollPitchYawrateThrust, self.attitudeRefCallback, queue_size=1)
        self.armPub = rospy.Publisher('/uav/input/arm', Empty, queue_size=2)
        self.resetPub = rospy.Publisher('/uav/input/reset', Empty, queue_size=2)
        # self.posePub = rospy.Publisher('/uav/command/pose', PoseStamped, queue_size=1)
        self.rpyRateThrustPub = rospy.Publisher('/uav/command/roll_pitch_yawrate_thrust', RollPitchYawrateThrust, queue_size=1)

        print('MPC tunner node started...')
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
        if self.start_saving_data and (not self.stop_saving_data):
            self.timeRollDect[msg.header.stamp.to_sec()] = euler[0]

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
        self.start_saving_data, self.stop_saving_data = False, False
        for rollGain in [40]:
            # reset the env
            self.resetPub.publish(Empty())
            # start the PID controller
            launch = self.startPIDController()
            launch.start()
            time.sleep(1)
            # start the dynamic configuration
            self.client = dynamic_reconfigure.client.Client("/uav/PID_attitude_controller", timeout=30, config_callback=self.dynamicReconfigureCallback)
            # update the configs
            self.client.update_configuration({"roll_gain":(rollGain)})
            # send command
            self.sendRPYrateThrust(roll=20*PI/180.0)
            self.start_saving_data = True
            self.rollErrorAbsSum = 0
            time.sleep(0.5)
            self.armPub.publish(Empty())
            rospy.sleep(4)
            self.stop_saving_data = True
            launch.shutdown()
            self.client.close()
            rospy.logwarn('rollGain={}, rollErrorAbsSum={}'.format(rollGain, self.rollErrorAbsSum) )
            time.sleep(1)
            self.plot_data()
        
    def plot_data(self):
        timeList = self.timeRollDect.keys()
        timeList.sort()
        rollList = []
        for t in timeList:
            rollList.append(self.timeRollDect[t])
        # rollList = self.timeRollDect.values()
        rollList = np.array(rollList, dtype=np.float64)*180.0/PI
        timeList = np.array(timeList, dtype=np.float64) - timeList[0]
        df = pd.DataFrame({'time': timeList, 'roll': rollList})
        df.to_pickle('./data_for_mpc_tunning/data_{}.pkl'.format(int(time.time())))
        plt.plot(timeList, rollList)
        plt.show()
        
        
            
    def dynamicReconfigureCallback(self, config):
        rospy.loginfo("Config set to {roll_gain}".format(**config))
    
    def spin(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            rate.sleep() 
    
        

def main():
    tunner = MPC_tunner()
    tunner.tune()
    # pidTunner.spin()



if __name__=='__main__':
	main()