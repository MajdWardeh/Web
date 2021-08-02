import rospy
import subprocess
import tf_conversions
from math import pi

def getQ(yaw):
    yaw = yaw * pi/180.0
    q = tf_conversions.transformations.quaternion_from_euler(0, 0, yaw)
    return [q[0], q[1], q[2], q[3]]

def main():
    x, y, z = (-10.422825041479353, 27, 1.67661692851683)
    sol = 10
    yaw = 90 + (23.5)
    q = getQ(yaw)    
    s1 = "rostopic pub -1 /uav/pose geometry_msgs/Pose  \"position:\n x: {0}\n y: {1}\n z: {2}\norientation:\n x: 0.0\n y: 0.0\n z: {3}\n w: {4}\"".format(x, y, z, q[2], q[3])
    subprocess.call("rostopic pub -1 /uav/pose geometry_msgs/Pose  \"position:\n x: {0}\n y: {1}\n z: {2}\norientation:\n x: 0.0\n y: 0.0\n z: {3}\n w: {4}\"".format(x, y, z, q[2], q[3]), shell=True, stdout=subprocess.PIPE)
    print(s1)


if __name__ == '__main__':
    main()