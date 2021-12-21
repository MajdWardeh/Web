import os
from math import degrees
from scipy.spatial.transform import Rotation as R
import rospy
from gazebo_msgs.msg import LinkStates



class GateListener:

	def __init__(self):
		rospy.init_node('gate_listener')
		linkStates_subs = rospy.Subscriber('/gazebo/link_states', LinkStates, self.linkStatesCallback)
		self.dict_init = False
		self.gatePosesDict = {}

	def linkStatesCallback(self, msg):
		statesNames = msg.name
		if not self.dict_init:
			self.dict_init = True
			for i, name in enumerate(statesNames):
				if name.startswith('gate'):
					gate_name = name.split('::')[0]
					self.gatePosesDict[gate_name] = msg.pose[i]

	def computeGatesLocation_FG_format(self):
		FG_dict = {}
		for name, pose in self.gatePosesDict.items():
			q = pose.orientation
			eulerAngles = R.from_quat([q.x, q.y, q.z, q.w]).as_euler('xyz', degrees=True)
			x = pose.position.x
			y = pose.position.y
			z = pose.position.z
			yaw = eulerAngles[-1]

			FG_dict[name] = [x, y, z, yaw]
		return FG_dict
			


	def saveToFile(self, dir):
		while not self.dict_init:
			rospy.sleep(0.5)
		rospy.sleep(0.2)
		print('gatePoseDict was initiated!')
		FG_dict = self.computeGatesLocation_FG_format()
		print(FG_dict)
		path = os.path.join(dir, 'gateLocationsFile.txt')
		with open(path, 'w+') as f:
			for i, gate in enumerate(FG_dict.values()):
				f.write('gate{}B: {}, {}, {}, {}, 2, 2, 2\n'.format(i, gate[0], gate[1], gate[2]-1.0, gate[3]))


def main():
	listener = GateListener()
	listener.saveToFile("/home/majd/drone_racing_ws/catkin_ddr/src/basic_rl_agent/scripts/environmentsCreation/from_sim2real")



if __name__ == '__main__':
	main()