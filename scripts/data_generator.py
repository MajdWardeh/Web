import os
import numpy as np
from numpy import linalg as la
import time
import subprocess
import shutil
from scipy.spatial.transform import Rotation
import math
import rospy
import tf
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, MultiArrayLayout
from geometry_msgs.msg import PoseStamped, Pose, Quaternion
from gazebo_msgs.msg import LinkStates
from mav_planning_msgs.msg import PolynomialTrajectory4D
from nav_msgs.msg import Path
from trajectory_msgs.msg import MultiDOFJointTrajectory
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from store_read_data import Data_Writer, Data_Reader

# from imutils import paths

# TODO:
# 1. modify the code to store data from multiple epochs.
# 2. sotre images on the go.
# 3. organize the placeAndSimulate function to reduce the delay due to 'sleep' commands.

class Dataset_collector:

    def __init__(self, camera_FPS=30, traj_length_per_image=2.1, numOfSamples=10):
        print("dataset collector started.")
        rospy.init_node('dataset_collector', anonymous=True)
        # rospy.Subscriber('/hummingbird/trajectory', PolynomialTrajectory4D, self.PolynomialTrajectoryCallback)
        rospy.Subscriber('/hummingbird/sampledTrajectoryChunk', Float64MultiArray, self.sampleTrajectoryChunkCallback, queue_size=50)
        # rospy.Subscriber('/hummingbird/command/trajectory', MultiDOFJointTrajectory, self.MultiDOFJointTrajectoryCallback)
        rospy.Subscriber('/hummingbird/rgb_camera/camera_1/image_raw', Image, self.rgbCameraCallback, queue_size=10)
        rospy.Subscriber('/gazebo/link_states', LinkStates, self.linkStatesCallback)
        self.sampleParticalTrajectory_pub = rospy.Publisher('/hummingbird/getTrajectoryChunk', Float64MultiArray, queue_size=1) 
        self.rvizPath_pub = rospy.Publisher('/path', Path, queue_size=10)
        self.bridge = CvBridge()
        self.camera_fps = camera_FPS
        self.traj_length_per_image = traj_length_per_image
        self.numOfSamples = numOfSamples 
        self.dt = (self.traj_length_per_image/self.camera_fps)/self.numOfSamples

        self.ts_rostime_index_dect = {} 
        self.ts_rostime_list = []
        self.imagesList = []
        self.numOfDataPoints = 100 
        self.numOfImageSequences = 4
        self.dataWriter = Data_Writer('first_data_collection', self.dt, self.numOfSamples, self.numOfDataPoints, (self.numOfImageSequences, 1))

        self.firstLinkStates = True
        self.STARTING_THRESH = 0.75 
        self.ENDING_THRESH = 1.25 
        self.epoch_finished = False

    def linkStatesCallback(self, msg):
        if self.firstLinkStates:
            self.firstLinkStates = False
            names = np.array(msg.name)
            self.robotIndex = np.argmax(names == 'hummingbird::hummingbird/base_link')
        x = msg.pose[self.robotIndex].position.x
        y = msg.pose[self.robotIndex].position.y
        z = msg.pose[self.robotIndex].position.z
        self.dronePoseition = np.array([x, y, z])
        
    # def PolynomialTrajectoryCallback(self, msg):
    #     print('this is a message: -----------------------------------------')
    #     # print(msg)
    
    def sampleTrajectoryChunkCallback(self, msg):
        data = np.array(msg.data)
        msg_ts_rostime = data[0]
        print('new msg received from sampleTrajectoryChunkCallback msg_ts_rostime={} --------------'.format(msg_ts_rostime))
        data = data[1:]
        data_length = data.shape[0]
        assert data_length==4*self.numOfSamples, "Error in the received message"
        if self.dataWriter.CanAddSample == True:
            msg_int_ts = int(msg_ts_rostime*1000) 
            msg_ts_index = self.ts_rostime_index_dect[msg_int_ts]  
            if msg_ts_index >= self.numOfImageSequences-1:
                Px, Py, Pz, Yaw = [], [], [], []
                for i in range(0, data.shape[0], 4):
                    Px.append(data[i])
                    Py.append(data[i+1])
                    Pz.append(data[i+2])
                    Yaw.append(data[i+3])
                    ts_rostime_sent = np.array(self.ts_rostime_list[msg_ts_index-4:msg_ts_index])*1000
                    ts_rostime_sent = ts_rostime_sent.astype(np.int64)
                    imageList_sent = [self.imagesList[msg_ts_index-4:msg_ts_index]]
                self.dataWriter.addSample(Px, Py, Pz, Yaw, imageList_sent, ts_rostime_sent)
        else:
            if self.dataWriter.data_saved == False:
                self.dataWriter.save_data()
                rospy.logwarn('data saved.....')
            rospy.logwarn('cannot add samples, the maximum number of samples is reached.')
        self.publishSampledPathRViz(data, msg_ts_rostime)
        

    def publishSampledPathRViz(self, data, msg_ts_rostime):
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

    # def MultiDOFJointTrajectoryCallback(self, msg):
    #     print('new MutliDOF massege')
    #     print(msg)

    def rgbCameraCallback(self, image_message):
        curr_drone_position = self.dronePoseition
        if self.epoch_finished == True:
            return
        if la.norm(curr_drone_position - self.droneStartingPosition) < self.STARTING_THRESH:
            rospy.logwarn("still not moved enough")
            return
        if la.norm(curr_drone_position - self.gatePosition) < self.ENDING_THRESH:
            rospy.logwarn("too close to the gate, epoch finished, saving data")
            self.epoch_finished = True
            self.dataWriter.save_data() 
            return
        # self.frame_counter += 1
        # # if self.frame_counter % 5 == 0:
        # if self.frame_counter < self.num_of_ignored_first_frames:
        #     return 
        ts_rostime = image_message.header.stamp.to_sec()
        cv_image = self.bridge.imgmsg_to_cv2(image_message, desired_encoding='passthrough')
        ts_id = int(ts_rostime*1000)
        self.ts_rostime_index_dect[ts_id] = len(self.imagesList)
        self.imagesList.append(cv_image)
        self.ts_rostime_list.append(ts_rostime)
        self.sendCommand(ts_rostime)

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
    
    def setGatePosition(self, gateX, gateY, gateZ):
        self.gatePosition = np.array([gateX, gateY, gateZ])
    
    def setDroneStartingPosition(self, droneX, droneY, droneZ):
        self.droneStartingPosition = np.array([droneX, droneY, droneZ])


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

def placeAndSimulate(data_collector):
    gateX, gateY, gateZ = 25.1521, 25.1935, 0.9
    gateWidth = 1.848
    ymin, ymax = (gateY-15), (gateY - 5) 
    rangeY = ymax - ymin
    y = np.random.exponential(0.7)
    y = ymin + min(y, rangeY)
    # y = ymin + np.random.rand() * (ymax-ymin)
    rangeX = (ymax - y)/(ymax - ymin) * (gateX * 0.3)
    xmin, xmax = gateX - rangeX, gateX + rangeX
    x = xmin + np.random.rand() * (xmax - xmin)
    droneX = x
    droneY = y
    droneZ = 1.5
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
        reference_frame: world }} }}\' &".format(droneX, droneY, droneZ, quat[2], quat[3]), shell=True)
    time.sleep(0.1)
    subprocess.call("roslaunch basic_rl_agent sample.launch &", shell=True)
    subprocess.call("rostopic pub -1 /hummingbird/command/pose geometry_msgs/PoseStamped \'{{header: {{stamp: now, frame_id: \"world\"}}, pose: {{position: {{x: {0}, y: {1}, z: {2}}}, orientation: {{z: {3}, w: {4} }} }} }}\' &".format(droneX, droneY, droneZ, quat[2], quat[3]), shell=True)
    data_collector.setGatePosition(gateX, gateY, gateZ)
    data_collector.setDroneStartingPosition(droneX, droneY, droneZ)
    time.sleep(0.5) 
    subprocess.call("rosservice call /gazebo/unpause_physics &", shell=True)
    time.sleep(0.2)
    subprocess.call("roslaunch basic_rl_agent plan_and_sample.launch &", shell=True)
    time.sleep(30)
    # print("hello from python-----------------------------------------------")
    
if __name__ == "__main__":
    collector = Dataset_collector()
    # r = rospy.Rate(1)
    # while not rospy.is_shutdown():
    #     main()
    #     r.sleep()
    # rospy.spin()
    for i in range(1):
        placeAndSimulate(collector)