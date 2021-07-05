import rospy
from quadrotor_msgs.msg import ControlCommand
from mav_msgs.msg import RateThrust

class CommandAdaptor:
    def __init__(self):
        rospy.init_node('control_command_adaptor_node', anonymous=True)
        self.controlCommandSub = rospy.Subscriber('/uav/control_command', ControlCommand, self.controlCommandCallback, queue_size=1)
        self.bodyRatesCommandPub = rospy.Publisher('/uav/input/rateThrust', RateThrust, queue_size=1)
        print('control command adaptor node started...')

    def controlCommandCallback(self, msg):
        # print(msg.expected_execution_time.to_sec()-rospy.Time.now().to_sec())
        rateThrustCommand = RateThrust()
        rateThrustCommand.header.stamp = rospy.Time.now()
        rateThrustCommand.angular_rates.x = msg.bodyrates.x
        rateThrustCommand.angular_rates.y = msg.bodyrates.y
        rateThrustCommand.angular_rates.z = msg.bodyrates.z
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