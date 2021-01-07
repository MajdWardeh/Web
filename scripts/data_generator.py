import os
import numpy as np
import time
import subprocess
import shutil
from scipy.spatial.transform import Rotation
import math
import rospy
import tf
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, MultiArrayLayout
from geometry_msgs.msg import PoseStamped, Pose, Quaternion
from mav_planning_msgs.msg import PolynomialTrajectory4D
from nav_msgs.msg import Path
from trajectory_msgs.msg import MultiDOFJointTrajectory
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# from imutils import paths

class Dataset_collector:

    def __init__(self, camera_FPS=30, traj_length_per_image=2.1, numOfSamples=10):
        print("dataset collector started.")
        rospy.init_node('dataset_collector', anonymous=True)
        # rospy.Subscriber('/hummingbird/trajectory', PolynomialTrajectory4D, self.PolynomialTrajectoryCallback)
        rospy.Subscriber('/hummingbird/sampledTrajectoryChunk', Float64MultiArray, self.sampleTrajectoryChunkCallback, queue_size=50)
        # rospy.Subscriber('/hummingbird/command/trajectory', MultiDOFJointTrajectory, self.MultiDOFJointTrajectoryCallback)
        rospy.Subscriber('/hummingbird/rgb_camera/camera_1/image_raw', Image, self.rgbCameraCallback, queue_size=10)
        self.sampleParticalTrajectory_pub = rospy.Publisher('/hummingbird/getTrajectoryChunk', Float64MultiArray, queue_size=1) 
        self.rvizPath_pub = rospy.Publisher('/path', Path, queue_size=10)
        self.bridge = CvBridge()
        self.camera_fps = camera_FPS
        self.traj_length_per_image = traj_length_per_image
        self.numOfSamples = numOfSamples
        self.dt = (self.traj_length_per_image/self.camera_fps)/self.numOfSamples

        #debug:
        self.frame_counter = 0
    # def PolynomialTrajectoryCallback(self, msg):
    #     print('this is a message: -----------------------------------------')
    #     # print(msg)
    
    def sampleTrajectoryChunkCallback(self, msg):
        print('new msg received from sampleTrajectoryChunkCallback --------------')
        data = np.array(msg.data)
        msg_ts_rostime = data[0]
        if msg_ts_rostime != self.current_ts_rostime:
            rospy.logwarn("the received trajectoryChunck msg was ignored; ts_rostime in the received msg %f does not match the curent_ts_rostime %f", msg_ts_rostime, self.current_ts_rostime)
            return
        data = data[1:]
        data_length = data.shape[0]
        assert data_length%4==0, "Error in the received message"
        poses_list = []
        for i in range(0, data.shape[0], 4):
            poseStamped_msg = PoseStamped()    
            poseStamped_msg.header.stamp = rospy.Time.from_sec(msg_ts_rostime + i*self.dt)
            poseStamped_msg.header.frame_id = 'world'
            poseStamped_msg.pose.position.x = data[i]
            poseStamped_msg.pose.position.y = data[i + 1]
            poseStamped_msg.pose.position.z = data[i + 2]
            quat = tf.transformations.quaternion_from_euler(0, 0, data[i+3])
            poseStamped_msg.pose.orientation.x = quat[0]
            poseStamped_msg.pose.orientation.y = quat[1]
            poseStamped_msg.pose.orientation.z = quat[2]
            poseStamped_msg.pose.orientation.w = quat[3]
            poses_list.append(poseStamped_msg)
        path = Path()
        path.poses = poses_list        
        path.header.stamp = rospy.get_rostime() #rospy.Time.from_sec(msg_ts_rostime)
        path.header.frame_id = 'world'
        self.rvizPath_pub.publish(path)

    def MultiDOFJointTrajectoryCallback(self, msg):
        print('new MutliDOF massege')
        # print(msg)

    def rgbCameraCallback(self, image_message):
        cv_image = self.bridge.imgmsg_to_cv2(image_message, desired_encoding='passthrough')
        self.frame_counter += 1
        if self.frame_counter % 5 == 0:
            ts_rostime = image_message.header.stamp.to_sec()
            self.current_ts_rostime = ts_rostime
            self.sendCommand(ts_rostime)

    def publishSampledPath(self):
        pass

    def sendCommand(self, ts_rostime):
        msg = Float64MultiArray()
        dim0 = MultiArrayDimension()
        dim0.label = 'ts_rostime, numOfsamples, dt'
        dim0.size = 3
        layout_var = MultiArrayLayout()
        layout_var.dim = [dim0]
        msg.layout = layout_var
        msg.data = [ts_rostime, self.numOfSamples, self.dt] 
        self.sampleParticalTrajectory_pub.publish(msg)

def GateInFieldOfView(gateX, gateY, gateWidth, cameraX, cameraY, cameraFOV, cameraRotation):
    r1 = (gateY - cameraY)/(gateX + gateWidth - cameraX)
    r2 = (gateY - cameraY)/(gateX - gateWidth - cameraX)
    print(math.atan(abs(r1)))
    print(math.atan(abs(r2)))
    print(cameraRotation - cameraFOV/2 + np.pi/2)
    print(cameraRotation + cameraFOV/2 + np.pi/2) 
    if math.atan(abs(r1)) > cameraFOV/2 and math.atan(abs(r2)) > cameraFOV/2:
        return True
    return False

def main():
    gateX, gateY = 25, 25
    gateWidth = 1.848
    ymin, ymax = (gateY-15), (gateY - 5) 
    rangeY = ymax - ymin
    y = np.random.exponential(0.7)
    y = ymin + min(y, rangeY)
    # y = ymin + np.random.rand() * (ymax-ymin)
    rangeX = (ymax - y)/(ymax - ymin) * (gateX * 0.3)
    xmin, xmax = gateX - rangeX, gateX + rangeX
    x = xmin + np.random.rand() * (xmax - xmin)
    # print("ymin: {}, ymax: {}, y: {}".format(ymin, ymax, y))
    # print("rangX: {}".format(rangeX))
    # print("xmin: {}, xmax: {}, x: {}".format(xmin, xmax, x))
    MaxCameraRotation = 30
    cameraRotation = np.random.normal(0, MaxCameraRotation/5) # 99.9% of the samples are in 5*segma
    #GateInFieldOfView(gateX, gateY, gateWidth, x, y, cameraFOV=1.5,  cameraRotation=cameraRotation*np.pi/180.0)
    #subprocess.call("roslaunch basic_rl_agent gazebo_only_trajectory_generation.launch &", shell=True)

    subprocess.call("rosnode kill /hummingbird/sampler &", shell=True)
    subprocess.call("rosservice call /gazebo/pause_physics &", shell=True)
    time.sleep(0.5)
    
    rot = Rotation.from_euler('z', cameraRotation + 90, degrees=True)
    quat = rot.as_quat()
    
    subprocess.call("rosservice call /gazebo/set_model_state \'{{model_state: {{ model_name: hummingbird, pose: {{ position: {{ x: {}, y: {} ,z: {} }},\
        orientation: {{x: 0, y: 0, z: {}, w: {} }} }}, twist:{{ linear: {{x: 0.0 , y: 0 ,z: 0 }} , angular: {{ x: 0.0 , y: 0 , z: 0.0 }} }}, \
        reference_frame: world }} }}\' &".format(x, y, 1.5, quat[2], quat[3]), shell=True)
    time.sleep(0.1)
    subprocess.call("roslaunch basic_rl_agent sample.launch &", shell=True)
    subprocess.call("rostopic pub -1 /hummingbird/command/pose geometry_msgs/PoseStamped \'{{header: {{stamp: now, frame_id: \"world\"}}, pose: {{position: {{x: {0}, y: {1}, z: {2}}}, orientation: {{z: {3}, w: {4} }} }} }}\' &".format(x, y, 1.5, quat[2], quat[3]), shell=True)
    time.sleep(0.5) 
    subprocess.call("rosservice call /gazebo/unpause_physics &", shell=True)
    time.sleep(0.2)
    subprocess.call("roslaunch basic_rl_agent plan_and_sample.launch &", shell=True)
    time.sleep(7)
    # print("hello from python-----------------------------------------------")
    
if __name__ == "__main__":
    collector = Dataset_collector()
    # r = rospy.Rate(1)
    # while not rospy.is_shutdown():
    #     main()
    #     r.sleep()
    # rospy.spin()
    for i in range(100):
        main()