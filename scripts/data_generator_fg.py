import os
import signal
import sys
import math
import numpy as np
from numpy import linalg as la
import time
import datetime
import subprocess
import shutil
from scipy.spatial.transform import Rotation
import math
from math import floor
import rospy
import roslaunch
import tf
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, MultiArrayLayout
from geometry_msgs.msg import PoseStamped, Pose, Quaternion
from gazebo_msgs.msg import ModelState, LinkStates
from mav_planning_msgs.msg import PolynomialTrajectory4D
from nav_msgs.msg import Path, Odometry
from trajectory_msgs.msg import MultiDOFJointTrajectory
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from store_read_data_extended import DataWriterExtended, DataReaderExtended

from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelState

# TODO:
#   1. collect a dataset with two images (stereo vision).
#   2. put folder for each run.
#   3. fps checker.
#   4. update the github.
#   5. random trajectories generation that keep the drone looking at the gate.



SAVE_DATA_DIR = '/home/majd/catkin_ws/src/basic_rl_agent/data/testing_data'
class Dataset_collector:

    def __init__(self, camera_FPS=30, traj_length_per_image=30.9, dt=-1, numOfSamples=120, numOfDatapointsInFile=500, save_data_dir=None, twist_data_length=100):
        rospy.init_node('dataset_collector', anonymous=True)
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
        ODOM_FREQUENCY = 100.0 # odometry frequency in Hz
        self.twist_data_len = twist_data_length # we want twist_data_length with the same frequency of the odometry
        self.twist_buff_maxSize = self.twist_data_len*20
        self.twist_tid_list = [] # stores the time as id from odometry msgs.
        self.twist_buff = [] # stores the samples from odometry coming at ODOM_FREQUENCY.

        # dataWriter flags:
        self.store_data = True 
        self.maxSamplesAchived = False

        # dataWriter stuff
        self.save_data_dir = save_data_dir
        if self.save_data_dir == None:
            self.save_data_dir = SAVE_DATA_DIR
        # create new directory for this run if store_data is True
        if self.store_data == True:
            self.save_data_dir = self.__createNewDirectory()
        self.dataWriter = self.__getNewDataWriter()

        self.STARTING_THRESH = 0.1 
        self.ENDING_THRESH = 1.55 #1.25 
        self.epoch_finished = False
        self.not_moving_counter = 0
        self.NOT_MOVING_THRES = 500
        self.droneStartingPosition_init = False
        self.gatePosition_init = False
       
        # Subscribers:
        self.sampledTrajectoryChunk_subs = rospy.Subscriber('/hummingbird/sampledTrajectoryChunk', Float64MultiArray, self.sampleTrajectoryChunkCallback, queue_size=50)
        self.camera_subs = rospy.Subscriber('/uav/camera/left_rgb_blurred/image_rect_color', Image, self.rgbCameraCallback, queue_size=2)
        self.odometry_subs = rospy.Subscriber('/hummingbird/ground_truth/odometry', Odometry, self.odometryCallback, queue_size=70)
        # Publishers:
        self.sampleParticalTrajectory_pub = rospy.Publisher('/hummingbird/getTrajectoryChunk', Float64MultiArray, queue_size=1) 
        self.rvizPath_pub = rospy.Publisher('/path', Path, queue_size=10)
        self.dronePosePub = rospy.Publisher('/hummingbird/command/pose', PoseStamped, queue_size=1)

        # print warning message if not storing data:
        if self.store_data == False:
            rospy.logwarn("store_data is False, data will not be saved...")

    def __createNewDirectory(self):
        dir_name = 'dataset_{}'.format(datetime.datetime.today().strftime('%Y%m%d%H%M_%S'))
        path = os.path.join(self.save_data_dir, dir_name)
        os.makedirs(path)
        return path

    def __getNewDataWriter(self):
        file_name = os.path.join(self.save_data_dir, 'data_{:d}'.format(int(round(time.time() * 1000))))
        return DataWriterExtended(file_name, self.dt, self.numOfSamples, self.numOfDataPoints, (self.numOfImageSequences, 1), (self.twist_data_len, 4) ) # the shape of each vel data sample is (twist_data_len, 4) because we have velocity on x,y,z and yaw

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
            # if ( (t_id - curr_tid_nparray[idx-self.twist_data_len+1:idx+1]) == np.arange(self.twist_data_len-1, -1, step=-1, dtype=np.int)).all(): # check the time sequence if it's equal to  (example) [5, 4, 3, 2, 1]
            return curr_twist_nparry[idx-self.twist_data_len+1:idx+1]
        return None

    def sampleTrajectoryChunkCallback(self, msg):
        data = np.array(msg.data)
        msg_ts_rostime = data[0]
        print('new msg received from sampleTrajectoryChunkCallback msg_ts_rostime={} --------------'.format(msg_ts_rostime))
        data = data[1:]
        data_length = data.shape[0]
        assert data_length==4*self.numOfSamples, "Error in the received message"
        if self.store_data:
            if self.dataWriter.CanAddSample() == True:
                msg_int_ts = int(msg_ts_rostime*1000) 
                msg_ts_index = self.ts_rostime_index_dect[msg_int_ts]  
                if msg_ts_index >= self.numOfImageSequences-1:
                    Px, Py, Pz, Yaw = [], [], [], []
                    for i in range(0, data.shape[0], 4):
                        # append the data to the variables:
                        Px.append(data[i])
                        Py.append(data[i+1])
                        Pz.append(data[i+2])
                        Yaw.append(data[i+3])

                        # take the images from the index: msg_ts_index [inclusive] back to the index: msg_ts_index-self.numOfImageSequences [exclusive]:
                        # first change the list to a numpy array and multipy all its values with 1000
                        ts_rostime_sent = np.array(self.ts_rostime_list[msg_ts_index-self.numOfImageSequences+1:msg_ts_index+1])*1000
                        # second, change its type to int
                        ts_rostime_sent = ts_rostime_sent.astype(np.int64)
                        
                        # imageList is the list of lists.
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

    def placeDrone(self, x, y, z, yaw=-1, qx=0, qy=0, qz=0, qw=0):
        # if yaw is provided (in degrees), then caculate the quaternion
        if yaw != -1:
            q = tf.transformations.quaternion_from_euler(0, 0, yaw*math.pi/180.0) 
            qx, qy, qz, qw = q[0], q[1], q[2], q[3]

        # send PoseStamp msg for the contorller:
        poseMsg = PoseStamped()
        poseMsg.header.stamp = rospy.Time.now()
        poseMsg.header.frame_id = 'hummingbird/base_link'
        poseMsg.pose.position.x = x
        poseMsg.pose.position.y = y
        poseMsg.pose.position.z = z
        poseMsg.pose.orientation.x = qx
        poseMsg.pose.orientation.y = qy
        poseMsg.pose.orientation.z = qz
        poseMsg.pose.orientation.w = qw
        self.dronePosePub.publish(poseMsg)

        # place the drone in gazebo using set_model_state service:
        state_msg = ModelState()
        state_msg.model_name = 'hummingbird'
        state_msg.pose.position.x = x 
        state_msg.pose.position.y = y
        state_msg.pose.position.z = z
        state_msg.pose.orientation.x = qx
        state_msg.pose.orientation.y = qy 
        state_msg.pose.orientation.z = qz 
        state_msg.pose.orientation.w = qw
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state(state_msg)
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e

        # update the the drone pose variables:
        self.setDroneStartingPosition(x, y, z)
        
    def pauseGazebo(self, pause=True):
        try:
            if pause:
                rospy.wait_for_service('/gazebo/pause_physics')
                pause_serv = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
                resp = pause_serv()
            else:
                rospy.wait_for_service('/gazebo/unpause_physics')
                unpause_serv = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
                resp = unpause_serv()
        except rospy.ServiceException, e:
            print('error while (un)pausing Gazebo')
            print(e)

    def __getPlanningLaunchObject(self):
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        launch = roslaunch.parent.ROSLaunchParent(uuid, ["/home/majd/catkin_ws/src/mav_trajectory_generation/mav_trajectory_generation_example/launch/flightGoggleEample.launch"], verbose=True)
        return launch

    def reset(self):
        # reset function deosn't clear maxSamplesAchived flag
        self.epoch_finished = False
        self.not_moving_counter = 0
        self.droneStartingPosition_init = False
        self.gatePosition_init = False
        # reset the variables of the imageCallback:
        self.ts_rostime_index_dect = {} 
        self.ts_rostime_list = []
        self.imagesList = []
        # reset the variables of the odomCallback:
        self.twist_tid_list = []
        self.twist_buff = []

    def __generateRandomPose(self, gateX, gateY, gateZ):
        xmin, xmax = gateX - 6, gateX - 15
        ymin, ymax = gateY - 3, gateY + 3
        zmin, zmax = gateZ - 1, gateZ + 2
        x = xmin + np.random.rand() * (xmax - xmin)
        y = ymin + np.random.rand() * (ymax - ymin)
        z = zmin + np.random.rand() * (zmax - zmin)

        # z_segma = 2/5
        # z = np.random.normal(gateZ, z_segma) 

        maxYawRotation = 25
        yaw = np.random.normal(0, maxYawRotation/5) # 99.9% of the samples are in 5*segma

        # x=15 #10
        # y=12 #7
        # z=3 #2.4
        # yaw=10
        return x, y, z, yaw

    def run(self, gateX=30.7, gateY=10, gateZ=2.4):
        for iteraction in range(400):
            # Place the drone:
            droneX, droneY, droneZ, droneYaw = self.__generateRandomPose(gateX, gateY, gateZ)
            self.placeDrone(droneX, droneY, droneZ, droneYaw)
            self.pauseGazebo()
            time.sleep(0.8)
            self.pauseGazebo(False)

            # set gate position:
            self.setGatePosition(gateX, gateY, gateZ)

            # Launch the planner:
            plannerLaunch = self.__getPlanningLaunchObject()
            plannerLaunch.start()

            # wait until the epoch finishs:
            rate = rospy.Rate(3)
            while not self.epoch_finished and not rospy.is_shutdown():
                rate.sleep()

            # shutdown the launch file:
            plannerLaunch.shutdown()

            # reset doesn't clear the maxSamplesAchived flag
            self.reset()
            rospy.sleep(1)

            # for each 4 iterations (episods), save data
            if iteraction % 4 == 0 and self.dataWriter.CanAddSample():
                self.dataWriter.save_data()
                self.maxSamplesAchived = True
                

            # if all samples are stored, get new dataWriter
            if self.maxSamplesAchived:
                self.dataWriter = self.__getNewDataWriter()
                self.maxSamplesAchived = False


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
    collector = Dataset_collector()
    time.sleep(3)
    # placeAndSimulate(collector)
    # llector.placeDrone(x=10, y=10, z=2.4, yaw=0)
    collector.run()



