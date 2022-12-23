#!/usr/bin/python
import os
from re import S
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
from flightgoggles.msg import IRMarkerArray
from mav_planning_msgs.msg import PolynomialTrajectory4D
from nav_msgs.msg import Path, Odometry
from trajectory_msgs.msg import MultiDOFJointTrajectory
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import dynamic_reconfigure.client
import cv2
from store_read_data_extended import DataWriterExtended, DataReaderExtended
from IrMarkersUtils import processMarkersMultiGate 

from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelState

# TODO:
#   1. collect a dataset with two images (stereo vision).
#   2. put folder for each run.
#   3. fps checker.
#   4. update the github.
#   5. random trajectories generation that keep the drone looking at the gate.



SAVE_DATA_DIR = '/home/majd/catkin_ws/src/basic_rl_agent/data2/flightgoggles/datasets/imageBezier_updated_datasets/imageBezierData_1000_30FPS'
class Dataset_collector:

    def __init__(self, camera_FPS=30, traj_length_per_image=30.9, dt=-1, numOfSamples=120, numOfDatapointsInFile=1000, save_data_dir=None, twist_data_length=500):
        rospy.init_node('dataset_collector', anonymous=True)
        self.camera_fps = camera_FPS
        self.traj_length_per_image = traj_length_per_image
        if dt == -1:
            self.numOfSamples = numOfSamples 
            self.dt = (self.traj_length_per_image/self.camera_fps)/self.numOfSamples
        else:
            self.dt = dt
            self.numOfSamples = (self.traj_length_per_image/self.camera_fps)/self.dt
        print('numOfSamplesPerTraj: {}, dt: {}, T: {}'.format(self.numOfSamples, self.dt, self.numOfSamples*self.dt))
        


        # RGB image callback variables
        self.imageShape = (240, 320, 3) # (h, w, ch)
        self.tid_image_dict = {} 
        self.image_tid_list = []
        self.imagesList = []
        self.numOfDataPoints = numOfDatapointsInFile 
        self.numOfImageSequence = 4
        self.bridge = CvBridge()
        self.cutting_tried = False

        # twist storage variables
        self.twist_data_len = twist_data_length # we want twist_data_length with the same frequency of the odometry
        self.twist_buff_maxSize = self.twist_data_len*30
        self.twist_tid_list = [] # stores the time as id from odometry msgs.
        self.twist_buff = [] # stores the samples from odometry coming at ODOM_FREQUENCY.

        ####################
        # dataWriter flags #
        ####################
        self.store_data = True # check SAVE_DATA_DIR
        self.store_markers = True
        self.store_images = True

        # dataWriter stuff
        self.save_data_dir = save_data_dir
        if self.save_data_dir == None:
            self.save_data_dir = SAVE_DATA_DIR
        # create new directory for this run if store_data is True
        if self.store_data == True:
            self.save_data_dir = self.__createNewDirectory()
        self.dataWriter = self.__getNewDataWriter()

        ###########################################
        #### Thresholds for collecting images  ####
        ###########################################
        self.STARTING_THRESH = 0.05
        self.ending_thresh = 1.25   
        self.cutting_thresh = 5
        self.TakeTheFirst10PerCent = False  # set dynamically in setGatePosition function 
        self.START_SKIPPING_THRESH = 5
        self.skipImages = 1

        self.DELTA_T_IMAGES = 2 # equals number of images to skip + 1
        self.IMAGE_TIME_DIFF = 0.016 * 1000 * self.DELTA_T_IMAGES # in [ms], 0.016 equals around 60FPS coming from FG simulator if Gazebo physices is set properly

        self.commands_sent_count = -1
        self.skip_sending_command_modulo = 1

        self.imageMsgsCounter = 0
        self.maxSamplesAchived = False
        self.epoch_finished = False
        self.not_moving_counter = 0
        self.NOT_MOVING_THRES = 500
        self.NOT_MOVING_SAMPLES = 5
        self.droneStartingPosition_init = False
        self.gatePosition_init = False

        # the location of the gate in FG V2.04 
        self.gate6CenterWorld = np.array([0.0, 0.0, 2.038498]).reshape(3, 1)

        # ir_beacons variables
        self.targetGate = 'gate0B'
        self.ts_rostime_markersData_dict = {}
        self.last_irMarker_stamp = None # used to print the time difference between consecutive images.

        ###### shutdown callback
        rospy.on_shutdown(self.shutdownCallback)
       
        # Subscribers:
        self.sampledTrajectoryChunk_subs = rospy.Subscriber('/hummingbird/sampledTrajectoryChunk', Float64MultiArray, self.sampleTrajectoryChunkCallback, queue_size=50)
        self.odometry_subs = rospy.Subscriber('/hummingbird/ground_truth/odometry', Odometry, self.odometryCallback, queue_size=100)
        self.camera_subs = rospy.Subscriber('/uav/camera/left/image_rect_color', Image, self.rgbCameraCallback, queue_size=2)
        self.markers_subs = rospy.Subscriber('/uav/camera/left/ir_beacons', IRMarkerArray, self.irMarkersCallback, queue_size=20)

        # Publishers:
        self.sampleParticalTrajectory_pub = rospy.Publisher('/hummingbird/getTrajectoryChunk', Float64MultiArray, queue_size=1) 
        self.rvizPath_pub = rospy.Publisher('/path', Path, queue_size=1)
        self.dronePosePub = rospy.Publisher('/hummingbird/command/pose', PoseStamped, queue_size=1)

        # print warning message if not storing data:
        if not self.store_data:
            rospy.logwarn("store_data is False, data will not be saved...")
        if not self.store_markers:
            rospy.logwarn("store_Markers is False")
        if not self.store_images:
            rospy.logwarn("store_image is False, images will not be saved...")
        
        if self.skip_sending_command_modulo != 1:
            rospy.logwarn("skip_sending_command_modulo = {}".format(self.skip_sending_command_modulo))
        

    def __createNewDirectory(self):
        dir_name = 'dataset_{}'.format(datetime.datetime.today().strftime('%Y%m%d%H%M_%S'))
        path = os.path.join(self.save_data_dir, dir_name)
        os.makedirs(path)
        return path

    def __getNewDataWriter(self):
        return DataWriterExtended(self.save_data_dir, self.dt, self.numOfSamples, self.numOfDataPoints, (self.numOfImageSequence, 1), (self.twist_data_len, 4), storeMarkers=self.store_markers, save_images_enabled=self.store_images) # the shape of each vel data sample is (twist_data_len, 4) because we have velocity on x,y,z and yaw

    def __del__(self):
        self.sampledTrajectoryChunk_subs.unregister() 
        self.camera_subs.unregister()
        self.odometry_subs.unregister()
        self.sampleParticalTrajectory_pub.unregister()
        self.rvizPath_pub.unregister()
        #del self.dataWriter
        print('destructor of the data_generator is called.')

    def shutdownCallback(self):
        if self.store_data:
            self.dataWriter.save_data()
    
    def odometryCallback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        self.dronePosition = np.array([x, y, z])
        twist = msg.twist.twist
        t_id = int(round(msg.header.stamp.to_sec()*1000))
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
        if (idx < curr_tid_nparray.shape[0]) and (curr_tid_nparray[idx] == t_id) \
            and (idx >= self.twist_data_len-1):
            # print('found, diff=', t_id-curr_tid_nparray[-1])
            return curr_twist_nparry[idx-self.twist_data_len+1:idx+1]
        # if idx == curr_tid_nparray.shape[0]:
        #     print('idx is out of range.........')
        #     print('diff=', t_id-curr_tid_nparray[-1])
        #     print(curr_tid_nparray[-1] - curr_tid_nparray[-10:])
        # print(idx < curr_tid_nparray.shape[0], (idx >= self.twist_data_len-1))
        return None

    def _get_tidList_forImageSequence(self, tid):
        '''
            @param: tid = int(round(ts_rostime*1000)) for an image.
            @return: a list of numbers correspond to the tid for each image that must be
                in the sequence which tid is last image in.
        '''
        curr_image_tid_array = np.array(self.image_tid_list)
        i = np.searchsorted(curr_image_tid_array, tid, side='left')

        start = i - self.DELTA_T_IMAGES * (self.numOfImageSequence - 1)
        end = i + 1
        if (i < curr_image_tid_array.shape[0]) and \
                (curr_image_tid_array[i] == tid) and (i >= start):
            ret_seq = curr_image_tid_array[start:end:self.DELTA_T_IMAGES]

            diff_seq = ret_seq[1:] - ret_seq[0:-1]
            diff_percent = np.abs((diff_seq - self.IMAGE_TIME_DIFF)/self.IMAGE_TIME_DIFF)
            # print('diff_seq=', diff_seq, ' ,diff_percent=', diff_percent, 'good? ', (diff_percent < 1.0).all())

            ## check if the timings among the consecutive images are correct
            correct_timing = (diff_percent < 0.1).all() 
            if not correct_timing:
                rospy.logwarn('incorrect timing. diff_seq: {}'.format(diff_seq))

            if ret_seq.shape[0] == self.numOfImageSequence and correct_timing:
                return ret_seq
        return None

    def sampleTrajectoryChunkCallback(self, msg):
        rospy.sleep(0.02)
        data = np.array(msg.data)
        msg_ts_rostime = data[0]
        print('new msg received from sampleTrajectoryChunkCallback msg_ts_rostime={} --------------'.format(msg_ts_rostime))
        data = data[1:]
        data_length = data.shape[0]
        assert data_length==4*self.numOfSamples, "Error in the received message"
        if self.store_data:
            if self.dataWriter.CanAddSample() == True:
                msg_tid = int(round(msg_ts_rostime*1000))
                Px, Py, Pz, Yaw = [], [], [], []
                for i in range(0, data.shape[0], 4):
                    # append the data to the variables:
                    Px.append(data[i])
                    Py.append(data[i+1])
                    Pz.append(data[i+2])
                    Yaw.append(data[i+3])

                # get a list of tid values for the sequcne that ends with the image defined by msg_tid
                tidList_imageSequence = self._get_tidList_forImageSequence(msg_tid)
                if tidList_imageSequence is None:
                    rospy.logwarn('tidList_imageSequence is None, returning...')
                    return
                
                # list of images to be saved
                imageList_sent = [[self.tid_image_dict[tid] for tid in tidList_imageSequence]]

                # processing markersData:
                markersDataList = []
                for tid in tidList_imageSequence:
                    if tid in self.ts_rostime_markersData_dict:
                        markersData = self.ts_rostime_markersData_dict[tid]
                    else:
                        rospy.logwarn('markersData for tid={} does not exist'.format(tid))
                        ts_rostime_markersData_dict_keys = self.ts_rostime_markersData_dict.keys() 
                        ts_rostime_markersData_dict_keys.sort()
                        print(ts_rostime_markersData_dict_keys[0:20])
                        markersData = np.zeros((4, 3))
                    markersDataList.append(markersData)

                # process twist data:
                twist_data_list = self._computeTwistDataList(msg_tid) 
                if twist_data_list is None:
                    rospy.logwarn('twist_data_list is None, returning...')
                    return

                # adding the sample
                if not self.store_markers:
                    self.dataWriter.addSample(Px, Py, Pz, Yaw, imageList_sent, tidList_imageSequence, twist_data_list)
                else:
                    self.dataWriter.addSample(Px, Py, Pz, Yaw, imageList_sent, tidList_imageSequence, twist_data_list, markersDataList)

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

    def irMarkersCallback(self, irMarkers_message):
        gatesMarkersDict = processMarkersMultiGate(irMarkers_message)
        if self.targetGate in gatesMarkersDict.keys():
            markersData = gatesMarkersDict[self.targetGate]
            tid = int(round(irMarkers_message.header.stamp.to_sec() * 1000))
            self.ts_rostime_markersData_dict[tid] = markersData

            ## printing time diff
            if self.last_irMarker_stamp != None:
                pass
                # print('time diff between two images:', (irMarkers_message.header.stamp.to_sec() - self.last_irMarker_stamp))
            self.last_irMarker_stamp = irMarkers_message.header.stamp.to_sec()


    def rgbCameraCallback(self, image_message):
        # must be computed as fast as possible:
        curr_drone_position = self.dronePosition


        cv_image = self.bridge.imgmsg_to_cv2(image_message, desired_encoding='bgr8')
        if cv_image.shape != self.imageShape:
            # rospy.logwarn('the received image size is different from what expected')
            cv_image = cv2.resize(cv_image, (self.imageShape[1], self.imageShape[0]))

        # take rostime stamps and images even though we might not send a command. They might be used for other sequences
        ts_rostime = image_message.header.stamp.to_sec()
        ts_id = int(round(ts_rostime*1000))
        self.image_tid_list.append(ts_id)
        self.tid_image_dict[ts_id] = cv_image
        self.imagesList.append(cv_image)

        # check if we will send command for this image (get its correspondance bezier trajector)
        if self.droneStartingPosition_init == False or self.gatePosition_init == False:
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
            if self.not_moving_counter >= self.NOT_MOVING_SAMPLES:
                return
        if la.norm(curr_drone_position - self.gatePosition) < self.ending_thresh:
            rospy.logwarn("too close to the gate, epoch finished")
            self.epoch_finished = True
            return


        ## skip the rest of the trajectory with prob
        # if la.norm(curr_drone_position - self.gatePosition) < self.cutting_thresh and not self.cutting_tried:
        #     prob = np.random.rand()
        #     if prob > 0.8 :
        #         rospy.logwarn("cutting thresh with {} prob occured, epoch finished".format(prob))
        #         self.epoch_finished = True
        #         return
        #     else:
        #         self.cutting_tried = True
        
        # skip images according to self.skipImages if the drone is close to the gate by self.START_SKIPPING_THRESH
        if la.norm(curr_drone_position - self.gatePosition) < self.START_SKIPPING_THRESH:
            self.imageMsgsCounter += 1
            if self.imageMsgsCounter % self.skipImages != 0:
                rospy.logwarn("skipped sending command")
                return

        self.commands_sent_count += 1
        if self.commands_sent_count % self.skip_sending_command_modulo != 0:
            return

        # send the command for this image
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
        # set the self.ending_thresh dynamiclly to stop the drone after small movement (this is in order to 
        # collect more starting data samples)
        if self.TakeTheFirst10PerCent:
            self.ending_thresh = la.norm(self.droneStartingPosition - self.gatePosition)*0.9
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
        except rospy.ServiceException as e:
            print("Service call failed: {}".format(e))

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
        except rospy.ServiceException as e:
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
        self.tid_image_dict = {} 
        self.image_tid_list = []
        self.imagesList = []
        self.cutting_tried = False
        self.commands_sent_count = -1
        # reset the variables of the odomCallback:
        self.twist_tid_list = []
        self.twist_buff = []
        # reset the variables related to ir_beacons
        self.ts_rostime_markersData_dict = {}
        self.last_irMarker_stamp = None

    def generateRandomPose(self, gateX, gateY, gateZ):
        xmin, xmax = gateX - 8, gateX + 8
        ymin, ymax = gateY - 15, gateY - 24
        zmin, zmax = gateZ - 1.0, gateZ + 2.0
        x = xmin + np.random.rand() * (xmax - xmin)
        y = ymin + np.random.rand() * (ymax - ymin)
        z = zmin + np.random.rand() * (zmax - zmin)
        # maxYawRotation = 55 #25
        # yaw = np.random.normal(90, maxYawRotation/5) # 99.9% of the samples are in 5*segma
        minYaw, maxYaw = 90-45, 90+45
        yaw = minYaw + np.random.rand() * (maxYaw - minYaw)
        return x, y, z, yaw

    
    def createTrajectoryConstraints(self):
        v0 = [0.0, -0.4, 2.03849800e+00, 1.570796327]
        v1 = [0.0, 7.0, 2.03849800e+00, 1.570796327]
        
        waypointsList = [v0, v1]

        ### writing the waypoints to file
        with open('/home/majd/catkin_ws/src/basic_rl_agent/scripts/environmentsCreation/txtFiles/posesLocations.yaml', 'w') as f:
            for i, v in enumerate(waypointsList):
                f.write('v{}: ['.format(i))
                for value in v:
                    if value != v[-1]:
                        f.write('{}, '.format(value))
                    else:
                        f.write('{}'.format(value))
                f.write(']\n')

    def run(self):
        gateX, gateY, gateZ = self.gate6CenterWorld.reshape(3, )
        self.createTrajectoryConstraints()
        for iteraction in range(10000):
            # Place the drone:
            droneX, droneY, droneZ, droneYaw = self.generateRandomPose(gateX, gateY, gateZ)
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
            rospy.sleep(1.5)

            # for each 2 iterations (episods), save data
            if iteraction % 2 == 0 and iteraction > 0 and self.store_data and self.dataWriter.CanAddSample():
                self.dataWriter.save_data()
                self.maxSamplesAchived = True
                

            # if all samples are stored, get a new dataWriter
            if self.maxSamplesAchived:
                self.dataWriter = self.__getNewDataWriter()
                self.maxSamplesAchived = False

# def dynamicReconfigureCallback(config):
#     rospy.loginfo("Config set to {time_step} and {max_update_rate}".format(**config))

def signal_handler(sig, frame):
    sys.exit(0)   

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    # update gazebo's physics
    # client = dynamic_reconfigure.client.Client("gazebo", timeout=30, config_callback=dynamicReconfigureCallback)
    # client.update_configuration({"max_update_rate": 1000.0, "time_step":0.001})
    # client.close()

    collector = Dataset_collector()
    time.sleep(3)
    # placeAndSimulate(collector)
    # llector.placeDrone(x=10, y=10, z=2.4, yaw=0)
    collector.run()



