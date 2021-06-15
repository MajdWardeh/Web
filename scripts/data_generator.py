import os
import signal
import sys
import numpy as np
from numpy import linalg as la
import time
import subprocess
import shutil
from scipy.spatial.transform import Rotation
import math
from math import floor
import rospy
import tf
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, MultiArrayLayout
from geometry_msgs.msg import PoseStamped, Pose, Quaternion
from gazebo_msgs.msg import LinkStates
from mav_planning_msgs.msg import PolynomialTrajectory4D
from nav_msgs.msg import Path, Odometry
from trajectory_msgs.msg import MultiDOFJointTrajectory
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from store_read_data_extended import DataWriterExtended, DataReaderExtended


# TODO:
# [IMPORTANT] sovle the bug of publishing on "self.rvizPath_pub.publish(path)" when the topic is closed. [done]
# solve the droneStartingPosition. [done]
# solve the unwanted movement of the drone when relocating it.

SAVE_DATA_DIR = '/home/majd/drone_racing_ws/catkin_ddr/src/basic_rl_agent/data/testing_data'
class Dataset_collector:

    def __init__(self, camera_FPS=30, traj_length_per_image=30.9, dt=-1, numOfSamples=120, numOfDatapointsInFile=200, save_data_dir=None, twist_data_length=5):
        rospy.init_node('dataset_collector', anonymous=True)
        # rospy.Subscriber('/gazebo/link_states', LinkStates, self.linkStatesCallback)
        self.firstOdometry = True
        self.bridge = CvBridge()
        self.camera_fps = camera_FPS
        self.traj_length_per_image = traj_length_per_image
        if dt == -1:
            self.numOfSamples = numOfSamples 
            self.dt = (self.traj_length_per_image/self.camera_fps)/self.numOfSamples
        else:
            self.dt = dt
            self.numOfSamples = (self.traj_length_per_image/self.camera_fps)/self.dt

        self.ts_rostime_index_dect = {} 
        self.ts_rostime_list = []
        self.imagesList = []
        self.numOfDataPoints = numOfDatapointsInFile 
        self.numOfImageSequences = 1
        # twist storage variables
        ODOM_FREQUENCY = 900.0 # odometry frequency in Hz
        self.twist_data_len = twist_data_length # we want twist_data_length with the same frequency of the odometry
        self.twist_buff_maxSize = self.twist_data_len*8 
        self.twist_tid_list = [] # stores the time as id from odometry msgs.
        self.twist_buff = [] # stores the samples from odometry coming at ODOM_FREQUENCY.

        if save_data_dir == None:
            save_data_dir = SAVE_DATA_DIR
        file_name = os.path.join(save_data_dir, 'data_{:d}'.format(int(round(time.time() * 1000))))
        self.dataWriter = DataWriterExtended(file_name, self.dt, self.numOfSamples, self.numOfDataPoints, (self.numOfImageSequences, 1), (self.twist_data_len, 4) ) # the shape of each vel data sample is (twist_data_len, 4) because we have velocity on x,y,z and yaw
        # debugging
        self.store_data = True 
        self.maxSamplesAchived = False

        self.STARTING_THRESH = 0.1 
        self.ENDING_THRESH = 1.25 
        self.epoch_finished = False
        self.not_moving_counter = 0
        self.NOT_MOVING_THRES = 500
        self.droneStartingPosition_init = False
        self.gatePosition_init = False
       
        # Subscribers and Publishers:
        self.sampledTrajectoryChunk_subs = rospy.Subscriber('/hummingbird/sampledTrajectoryChunk', Float64MultiArray, self.sampleTrajectoryChunkCallback, queue_size=50)
        self.camera_subs = rospy.Subscriber('/hummingbird/rgb_camera/camera_1/image_raw', Image, self.rgbCameraCallback, queue_size=2)
        self.odometry_subs = rospy.Subscriber('/hummingbird/ground_truth/odometry', Odometry, self.odometryCallback, queue_size=70)
        self.sampleParticalTrajectory_pub = rospy.Publisher('/hummingbird/getTrajectoryChunk', Float64MultiArray, queue_size=1) 
        self.rvizPath_pub = rospy.Publisher('/path', Path, queue_size=10)
        # print warning message if not storing data:
        if self.store_data == False:
            rospy.logwarn("store_data is False, data will not be saved...")

    def __del__(self):
        self.sampledTrajectoryChunk_subs.unregister() 
        self.camera_subs.unregister()
        self.odometry_subs.unregister()
        self.sampleParticalTrajectory_pub.unregister()
        self.rvizPath_pub.unregister()
        del self.dataWriter
        print('destructor of the data_generator is called.')
      
    def delete(self):
        self.__del__()
    
    # def linkStatesCallback(self, msg):
        # if self.firstOdometry:
        #     self.firstOdometry = False 
        #     names = np.array(msg.name)
        #     self.robotIndex = np.argmax(names == 'hummingbird::hummingbird/base_link')
        # x = msg.pose[self.robotIndex].position.x
        # y = msg.pose[self.robotIndex].position.y
        # z = msg.pose[self.robotIndex].position.z
        # self.dronePosition = np.array([x, y, z])
    
    def odometryCallback(self, msg):
        self.firstOdometry = False
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        self.dronePosition = np.array([x, y, z])
        twist = msg.twist.twist
        t_id = int(msg.header.stamp.to_sec()*1000)
        twist_data = np.array([twist.linear.x, twist.linear.y, twist.linear.z, twist.angular.z])
        self.twist_tid_list.append(t_id)
        self.twist_buff.append(twist_data)
        if len(self.twist_buff) > self.twist_buff_maxSize:
            self.twist_buff = self.twist_buff[-self.twist_buff_maxSize :]
            self.twist_tid_list = self.twist_tid_list[-self.twist_buff_maxSize :]
            

    def _computeTwistDataList(self, t_id):
        curr_tid_nparray  = np.array(self.twist_tid_list)
        curr_twist_nparry = np.array(self.twist_buff)
        idx = np.searchsorted(curr_tid_nparray, t_id, side='left')
        # check if idx is not out of range or is not the last element in the array (there is no upper bound)
        # take the data from the idx [inclusive] back to idx-self.twist_data_len [exclusive]
        if idx <= self.twist_buff_maxSize-2 and idx-self.twist_data_len+1>= 0:
            if ( (t_id - curr_tid_nparray[idx-self.twist_data_len+1:idx+1]) == np.arange(self.twist_data_len-1, -1, step=-1, dtype=np.int)).all():
                return curr_twist_nparry[idx-self.twist_data_len+1:idx+1]
            else:
                print('condition on Twist data failed, sample will not be considered.')
        return None


    def sampleTrajectoryChunkCallback(self, msg):
        data = np.array(msg.data)
        msg_ts_rostime = data[0]
        print('new msg received from sampleTrajectoryChunkCallback msg_ts_rostime={} --------------'.format(msg_ts_rostime))
        data = data[1:]
        data_length = data.shape[0]
        assert data_length==4*self.numOfSamples, "Error in the received message"
        if self.store_data:
            rospy.logerr("dataWriter.CanAddSample = {}, index={}, file_name={}".format(self.dataWriter.CanAddSample(), self.dataWriter.getIndex(), self.dataWriter.file_name))
            
            if self.dataWriter.CanAddSample() == True:
                msg_int_ts = int(msg_ts_rostime*1000) 
                msg_ts_index = self.ts_rostime_index_dect[msg_int_ts]  
                if msg_ts_index >= self.numOfImageSequences-1:
                    Px, Py, Pz, Yaw = [], [], [], []
                    for i in range(0, data.shape[0], 4):
                        Px.append(data[i])
                        Py.append(data[i+1])
                        Pz.append(data[i+2])
                        Yaw.append(data[i+3])
                        # take the images from the index: msg_ts_index [inclusive] back to the index: msg_ts_index-self.numOfImageSequences [exclusive].
                        ts_rostime_sent = np.array(self.ts_rostime_list[msg_ts_index-self.numOfImageSequences+1:msg_ts_index+1], dtype=np.int64)*1000
                        imageList_sent = [self.imagesList[msg_ts_index-self.numOfImageSequences+1:msg_ts_index+1]]
                    # process twist data:
                    twist_data_list = self._computeTwistDataList(msg_int_ts) 
                    if twist_data_list is None:
                        rospy.logwarn('twist_data_list is None, returning...')
                        return
                    # adding the sample
                    self.dataWriter.addSample(Px, Py, Pz, Yaw, imageList_sent, ts_rostime_sent, twist_data_list)
            else:
                if self.dataWriter.data_saved == False:
                    self.dataWriter.save_data()
                    rospy.logwarn('data saved.....')
                self.maxSamplesAchived = True
                self.epoch_finished = True
                rospy.logwarn('cannot add samples, the maximum number of samples is reached.')
        try:
            self.publishSampledPathRViz(data, msg_ts_rostime)
        except:
            pass
        

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


    def rgbCameraCallback(self, image_message):
        # must be computed as fast as possible:
        curr_drone_position = self.dronePosition
        # rest of the code:
        if self.droneStartingPosition_init == False or self.gatePosition_init == False or self.firstOdometry == True:
            return
        if self.epoch_finished == True:
            return
        if la.norm(curr_drone_position - self.droneStartingPosition) < self.STARTING_THRESH:
            if self.not_moving_counter == 0:
                rospy.logwarn("still not moved enough")
            self.not_moving_counter += 1
            if self.not_moving_counter >= self.NOT_MOVING_THRES:
                self.epoch_finished = True
                rospy.logwarn("did not move, time out, epoch finished")
            return
        if la.norm(curr_drone_position - self.gatePosition) < self.ENDING_THRESH:
            rospy.logwarn("too close to the gate, epoch finished")
            self.epoch_finished = True
            return
        ts_rostime = image_message.header.stamp.to_sec()
        cv_image = self.bridge.imgmsg_to_cv2(image_message, desired_encoding='bgr8')
        ts_id = int(ts_rostime*1000)
        self.ts_rostime_index_dect[ts_id] = len(self.imagesList) # a mapping from ts_rostime to image index in the imagesList
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
        self.gatePosition_init = True
    
    def setDroneStartingPosition(self, droneX, droneY, droneZ):
        self.droneStartingPosition = np.array([droneX, droneY, droneZ])
        self.droneStartingPosition_init = True
    
    def reset(self):
        self.epoch_finished = False
        self.not_moving_counter = 0
        self.droneStartingPosition_init = False
        self.gatePosition_init = False

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
    # y = np.random.exponential(0.7)
    # y = ymin + min(y, rangeY)
    y = ymin + np.random.rand() * (ymax-ymin)
    rangeX = (ymax - y)/(ymax - ymin) * (gateX * 0.3)
    xmin, xmax = gateX - rangeX, gateX + rangeX
    x = xmin + np.random.rand() * (xmax - xmin)
    droneX = x
    droneY = y
    droneZ = np.random.normal(1.5, 0.2) 
    # print("ymin: {}, ymax: {}, y: {}".format(ymin, ymax, y))
    # print("rangX: {}".format(rangeX))
    # print("xmin: {}, xmax: {}, x: {}".format(xmin, xmax, x))
    MaxCameraRotation = 30
    cameraRotation = np.random.normal(0, MaxCameraRotation/5) # 99.9% of the samples are in 5*segma
    #GateInFieldOfView(gateX, gateY, gateWidth, x, y, cameraFOV=1.5,  cameraRotation=cameraRotation*np.pi/180.0)


    time.sleep(0.5)
    subprocess.call("rosnode kill /hummingbird/sampler &", shell=True, stdout=subprocess.PIPE)
    time.sleep(3)
    subprocess.call("rosservice call /gazebo/pause_physics &", shell=True, stdout=subprocess.PIPE)
    time.sleep(0.5)
    
    rot = Rotation.from_euler('z', cameraRotation + 90, degrees=True)
    quat = rot.as_quat()
    
    subprocess.call("rosservice call /gazebo/set_model_state \'{{model_state: {{ model_name: hummingbird, pose: {{ position: {{ x: {}, y: {} ,z: {} }},\
        orientation: {{x: 0, y: 0, z: {}, w: {} }} }}, twist:{{ linear: {{x: 0.0 , y: 0 ,z: 0 }} , angular: {{ x: 0.0 , y: 0 , z: 0.0 }} }}, \
        reference_frame: world }} }}\' &".format(droneX, droneY, droneZ, quat[2], quat[3]), shell=True, stdout=subprocess.PIPE)

    subprocess.call("roslaunch basic_rl_agent sample.launch &", shell=True, stdout=subprocess.PIPE)
    subprocess.call("rostopic pub -1 /hummingbird/command/pose geometry_msgs/PoseStamped \'{{header: {{stamp: now, frame_id: \"world\"}}, pose: {{position: {{x: {0}, y: {1}, z: {2}}}, orientation: {{z: {3}, w: {4} }} }} }}\' &".format(droneX, droneY, droneZ, quat[2], quat[3]), shell=True, stdout=subprocess.PIPE)
    data_collector.reset()
    data_collector.setGatePosition(gateX, gateY, gateZ)
    data_collector.setDroneStartingPosition(droneX, droneY, droneZ)
    time.sleep(1.5) 
    subprocess.call("rosservice call /gazebo/unpause_physics &", shell=True, stdout=subprocess.PIPE)
    time.sleep(0.5)
    subprocess.call("roslaunch basic_rl_agent plan_and_sample.launch &", shell=True, stdout=subprocess.PIPE)
    while not data_collector.epoch_finished:
        time.sleep(0.2)

def signal_handler(sig, frame):
    sys.exit(0)   

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    for epoch in range(1):
        print("-----------------------------------------------")
        print("Epoch: #{}".format(epoch))
        print("-----------------------------------------------")
        collector = Dataset_collector()
        while not collector.maxSamplesAchived:
            print("new place and simulate, maxSamplesAchived={}".format(collector.maxSamplesAchived))
            placeAndSimulate(collector)
    print("done.")


