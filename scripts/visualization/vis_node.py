import math
import rospy
from visualization_msgs.msg import Marker


def main():
    rospy.init_node('vis_node', anonymous=True)
    gatePub = rospy.Publisher('/gate_marker', Marker, queue_size=1)
    markerMsg = createMarkerMsg()

    rate  = rospy.Rate(10)
    while not rospy.is_shutdown():
        gatePub.publish(markerMsg)
        rate.sleep()


def createMarkerMsg():
    marker = Marker()
    marker.header.frame_id = "world"
    marker.header.stamp = rospy.Time()
    marker.ns = "my_namespace"
    marker.id = 0
    marker.type = Marker.MESH_RESOURCE
    marker.action = Marker.ADD
    marker.pose.position.x = 0
    marker.pose.position.y = 0
    marker.pose.position.z = 0.5
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0 #-math.sqrt(2)
    marker.pose.orientation.w = 1.0 #math.sqrt(2)
    marker.scale.x = 0.012
    marker.scale.y = 0.012
    marker.scale.z = 0.012
    marker.color.a = 1.0 # Don't forget to set the alpha!
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.mesh_resource = "file:///home/majd/Desktop/gate.dae"
    return marker


if __name__ == "__main__":
    main()