import rospy

from mav_msgs.msg import RateThrust, RollPitchYawrateThrust

class CommandAdaptor:
    def __init__(self):
        rospy.init_node('control_command_adaptor_node', anonymous=True)
        self.controlCommandSub = rospy.Subscriber('/uav/command/roll_pitch_yawrate_thrust', RollPitchYawrateThrust, self.rollPitchYawrateThrustCallback, queue_size=1)
        self.bodyRatesCommandPub = rospy.Publisher('/uav/input/rateThrust', RateThrust, queue_size=1)
        print('control command adaptor node started...')

    def rollPitchYawrateThrustCallback(self, msg):
        # print(msg.expected_execution_time.to_sec()-rospy.Time.now().to_sec())
        rateThrustCommand = RateThrust()
        rateThrustCommand.header.stamp = msg.header.stamp
        rateThrustCommand.angular_rates.x = msg.roll
        rateThrustCommand.angular_rates.y = msg.pitch
        rateThrustCommand.angular_rates.z = msg.yaw_rate
        rateThrustCommand.thrust.x = 0
        rateThrustCommand.thrust.y = 0
        rateThrustCommand.thrust.z = msg.collective_thrust
        self.bodyRatesCommandPub.publish(rateThrustCommand)
        print(rateThrustCommand)
       

    def spin(self):
        rospy.spin()


def main():
    adaptor = CommandAdaptor()
    adaptor.spin()


if __name__=='__main__':
	main()