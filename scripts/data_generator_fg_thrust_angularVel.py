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
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, MultiArrayLayout, Empty, Bool
from geometry_msgs.msg import PoseStamped, Pose, Quaternion
from gazebo_msgs.msg import ModelState, LinkStates
from quadrotor_msgs.msg import ControlCommand
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

from std_srvs.srv import Empty as Empty_srv
from gazebo_msgs.srv import SetModelState

# TODO:
#   1. collect a dataset with two images (stereo vision).
#   2. put folder for each run.
#   3. fps checker.
#   4. update the github.
#   5. random trajectories generation that keep the drone looking at the gate.



SAVE_DATA_DIR = '/home/majd/catkin_ws/src/basic_rl_agent/data2/flightgoggles/datasets/imageLowLevelControl_1000'
class Dataset_collector:

    def __init__(self, camera_FPS=30, traj_length_per_image=30.9, dt=-1, numOfSamples=100, numOfDatapointsInFile=1500, save_data_dir=None, twist_data_length=100):
        rospy.init_node('dataset_collector', anonymous=True)
        self.camera_fps = camera_FPS
        self.traj_length_per_image = traj_length_per_image
        if dt == -1:
            self.numOfSamples = numOfSamples 
            self.dt = (self.traj_length_per_image/self.camera_fps)/self.numOfSamples
        else:
            self.dt = dt
            self.numOfSamples = (self.traj_length_per_image/self.camera_fps)/self.dt

        # RGB image callback variables
        self.imageShape = (240, 320, 3) # (h, w, ch)
        self.tid_image_dict = {} 
        self.image_tid_list = []
        self.imagesList = []
        self.numOfDataPoints = numOfDatapointsInFile 
        self.numOfImageSequence = 5
        self.bridge = CvBridge()

        # twist storage variables
        self.twist_data_len = twist_data_length # we want twist_data_length with the same frequency of the odometry
        self.twist_buff_maxSize = self.twist_data_len*30
        self.twist_tid_list = [] # stores the time as id from odometry msgs.
        self.twist_buff = [] # stores the samples from odometry coming at ODOM_FREQUENCY.

        # low-level control commands variabls:
        self.controlCommand_tid_dict = {}
        self.COMMAND_LENGTH = 100 # for 1 second
        self.COMMAND_STAMP_DIFF_THRESH = 20

        self.tid_samplesToSave_list = []


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
        self.TakeTheFirst10PerCent = False  # set dynamically in setGatePosition function 
        self.START_SKIPPING_THRESH = 5
        self.skipImages = 1

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

        ###### shutdown callback
        rospy.on_shutdown(self.shutdownCallback)
       
        # Subscribers:
        self.lowLevelControlCommand_subs = rospy.Subscriber('/hummingbird/control_command', ControlCommand, self.lowLevelControlCommandCallback, queue_size=100)
        self.odometry_subs = rospy.Subscriber('/hummingbird/ground_truth/odometry', Odometry, self.odometryCallback, queue_size=100)
        self.camera_subs = rospy.Subscriber('/uav/camera/left/image_rect_color', Image, self.rgbCameraCallback, queue_size=2)
        self.markers_subs = rospy.Subscriber('/uav/camera/left/ir_beacons', IRMarkerArray, self.irMarkersCallback, queue_size=20)

        # Publishers:
        self.rvizPath_pub = rospy.Publisher('/path', Path, queue_size=1)
        self.dronePosePub = rospy.Publisher('/hummingbird/autopilot/pose_command', PoseStamped, queue_size=1)
        self.drone_forceHover_pub = rospy.Publisher('/hummingbird/autopilot/force_hover', Empty, queue_size=1)
        self.drone_startController_pub = rospy.Publisher('/hummingbird/autopilot/start', Empty, queue_size=1)
        self.drone_arm = rospy.Publisher('/hummingbird/bridge/arm', Bool, queue_size=1)
        


        # print warning message if not storing data:
        if not self.store_data:
            rospy.logwarn("store_data is False, data will not be saved...")
        if not self.store_markers:
            rospy.logwarn("store_Markers is False")
        

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
            @param: tid = int(ts_rostime)*1000 for an image.
            @return: a list of numbers correspond to the tid for each image that must be
                in the sequence which tid is last image in.
        '''
        curr_image_tid_array = np.array(self.image_tid_list)
        i = np.searchsorted(curr_image_tid_array, tid, side='left')

        if (curr_image_tid_array[i] == tid) and (i >= self.numOfImageSequence-1):
            # print('found, diff=', tid-curr_image_tid_array[-1])
            return curr_image_tid_array[i-self.numOfImageSequence+1:i+1]
        # print('idx is out of range.........')
        # print('diff=', tid-curr_image_tid_array[-1])
        # print(curr_image_tid_array[-1] - curr_image_tid_array[-10:])
        # print(curr_image_tid_array[i] == tid, i >= self.numOfImageSequence-1)
        return None

    def lowLevelControlCommandCallback(self, msg):
        t_id = int(msg.header.stamp.to_sec()*1000)
        command = [msg.collective_thrust, msg.bodyrates.x, msg.bodyrates.y, msg.bodyrates.z]
        self.controlCommand_tid_dict[t_id] = command

    def __computeLowLevelCommandList(self, image_tid):
        ti_list = np.array(self.controlCommand_tid_dict.keys())
        ti_list = np.sort(ti_list)
        idx = np.searchsorted(ti_list, image_tid, side='left')
        if (idx + self.COMMAND_LENGTH) < ti_list.shape[0]:
            # checking stamps differences:
            l1 = ti_list[ idx : idx+self.COMMAND_LENGTH]
            l2 = ti_list[ idx+1 : idx+self.COMMAND_LENGTH+1]
            max_stamp_diff = (l2-l1).max()
            if max_stamp_diff > self.COMMAND_STAMP_DIFF_THRESH:
                rospy.logwarn('found max_stamp_diff: {} larger than expected: {}, returning None'.format(max_stamp_diff, self.COMMAND_STAMP_DIFF_THRESH))
                return None

            commandList = []
            for t in l1:
                command = self.controlCommand_tid_dict[t]
                if command is None:
                    rospy.logwarn('found a None command in  controlCommand_tid_dict at t:{}'.format(t))
                    return None
                commandList.append(command)

            assert len(commandList) == self.COMMAND_LENGTH
            return commandList
        rospy.logwarn('idx: {} was not found for image_tid: {}'.format(idx, image_tid))
        return None
            

    def saveDataSample(self, msg_tid):
        if self.store_data:
            if self.dataWriter.CanAddSample() == True:
                commandList = self.__computeLowLevelCommandList(msg_tid)
                if commandList is None:
                    rospy.logwarn('commandList is None, returning...')
                    return
                Pthrust, Px, Py, Pz = [], [], [], []
                for command in commandList:
                    # append the data to the variables:
                    assert len(command) == 4
                    Pthrust.append(command[0])
                    Px.append(command[1])
                    Py.append(command[2])
                    Pz.append(command[3])

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
                        rospy.logwarn('markersData for tid={} does not exist')
                        markersData = np.zeros((4, 3))
                    markersDataList.append(markersData)

                # process twist data:
                twist_data_list = self._computeTwistDataList(msg_tid) 
                if twist_data_list is None:
                    rospy.logwarn('twist_data_list is None, returning...')
                    return

                # adding the sample
                if not self.store_markers:
                    self.dataWriter.addSample(Pthrust, Px, Py, Pz, imageList_sent, tidList_imageSequence, twist_data_list)
                else:
                    self.dataWriter.addSample(Pthrust, Px, Py, Pz, imageList_sent, tidList_imageSequence, twist_data_list, markersDataList)
                print('saved data sample msg_tid={} .......'.format(msg_tid))

            else:
                if self.dataWriter.data_saved == False:
                    self.dataWriter.save_data()
                    rospy.logwarn('data saved.....')
                self.maxSamplesAchived = True
                self.epoch_finished = True
                rospy.logwarn('cannot add samples, the maximum number of samples is reached.')
        
    def saveCollectedSamples(self):
        for image_tid in self.tid_samplesToSave_list:
            markersData = self.ts_rostime_markersData_dict.get(image_tid, None)
            if markersData is None:
                rospy.logwarn('sampled: tid {} was not found'.format(image_tid))
                continue
            self.saveDataSample(image_tid)
            # image = self.tid_image_dict[image_tid]
            # cv2.imshow('image', image)
            # cv2.waitKey(0)

    def irMarkersCallback(self, irMarkers_message):
        gatesMarkersDict = processMarkersMultiGate(irMarkers_message)
        if self.targetGate in gatesMarkersDict.keys():
            markersData = gatesMarkersDict[self.targetGate]
            tid = int(irMarkers_message.header.stamp.to_sec() * 1000)
            self.ts_rostime_markersData_dict[tid] = markersData

    def rgbCameraCallback(self, image_message):
        # must be computed as fast as possible:
        curr_drone_position = self.dronePosition

        cv_image = self.bridge.imgmsg_to_cv2(image_message, desired_encoding='bgr8')
        if cv_image.shape != self.imageShape:
            # rospy.logwarn('the received image size is different from what expected')
            cv_image = cv2.resize(cv_image, (self.imageShape[1], self.imageShape[0]))

        # take rostime stamps and images even though we might not send a command. They might be used for other sequences
        ts_rostime = image_message.header.stamp.to_sec()
        ts_id = int(ts_rostime*1000)
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
        # skip images according to self.skipImages if the drone is close to the gate by self.START_SKIPPING_THRESH
        if la.norm(curr_drone_position - self.gatePosition) < self.START_SKIPPING_THRESH:
            self.imageMsgsCounter += 1
            if self.imageMsgsCounter % self.skipImages != 0:
                return

        # save the tid for this image
        self.tid_samplesToSave_list.append(ts_id)

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

        for _ in range(10):
            self.drone_forceHover_pub.publish(Empty())
            self.dronePosePub.publish(poseMsg)
            rospy.sleep(0.1)
        
    def pauseGazebo(self, pause=True):
        try:
            if pause:
                rospy.wait_for_service('/gazebo/pause_physics')
                pause_serv = rospy.ServiceProxy('/gazebo/pause_physics', Empty_srv)
                resp = pause_serv()
            else:
                rospy.wait_for_service('/gazebo/unpause_physics')
                unpause_serv = rospy.ServiceProxy('/gazebo/unpause_physics', Empty_srv)
                resp = unpause_serv()
        except rospy.ServiceException as e:
            print('error while (un)pausing Gazebo')
            print(e)

    def __getPlanningLaunchObject(self):
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        launch = roslaunch.parent.ROSLaunchParent(uuid, ["/home/majd/catkin_ws/src/mav_trajectory_generation/mav_trajectory_generation_example/launch/flightGoggleEample.launch"], verbose=True)
        return launch

    def __getControllerLaunchObject(self):
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        launch = roslaunch.parent.ROSLaunchParent(uuid, ["/home/majd/catkin1_ws/src/Flightgoggles/flightgoggles/launch/rpg_controller_only.launch"], verbose=True)
        return launch

    def launch_new_controller(self):
        controllerLaunch = self.__getControllerLaunchObject()
        time.sleep(1.)
        controllerLaunch.start()
        self.drone_arm.publish(Bool(True))
        time.sleep(0.5)
        self.drone_startController_pub.publish(Empty())
        self.drone_forceHover_pub.publish(Empty())
        return controllerLaunch


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
        # reset the variables of the odomCallback:
        self.twist_tid_list = []
        self.twist_buff = []
        # reset the variables related to ir_beacons
        self.ts_rostime_markersData_dict = {}
        # reset the low-level control commands variables:
        self.controlCommand_tid_dict = {}

        self.tid_samplesToSave_list = []


    def generateRandomPose(self, gateX, gateY, gateZ):
        xmin, xmax = gateX - 8, gateX + 8
        ymin, ymax = gateY - 15, gateY - 24
        zmin, zmax = gateZ - 1.0, gateZ + 2.0
        x = xmin + np.random.rand() * (xmax - xmin)
        y = ymin + np.random.rand() * (ymax - ymin)
        z = zmin + np.random.rand() * (zmax - zmin)
        # maxYawRotation = 55 #25
        # yaw = np.random.normal(90, maxYawRotation/5) # 99.9% of the samples are in 5*segma
        minYaw, maxYaw = 90-40, 90+40
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
        timeOut_thresh = 60
        timeOut = False
        controllerLaunch = self.launch_new_controller()

        # self.createTrajectoryConstraints()
        for iteraction in range(100000):
            if timeOut == True:
                timeOut = False
                controllerLaunch = self.launch_new_controller()
            
            # Place the drone:
            droneX, droneY, droneZ, droneYaw = self.generateRandomPose(gateX, gateY, gateZ)
            self.placeDrone(droneX, droneY, droneZ, droneYaw)
            time.sleep(2.2)

            # set gate position:
            self.setGatePosition(gateX, gateY, gateZ)

            # Launch the planner:
            plannerLaunch = self.__getPlanningLaunchObject()
            plannerLaunch.start()

            # wait until the epoch finishs:
            rate = rospy.Rate(3)
            ts = time.time()
            while not self.epoch_finished and not rospy.is_shutdown():
                te = time.time()
                if time.time() - ts > timeOut_thresh:
                    self.epoch_finished = True
                    timeOut = True
                rate.sleep()

            # shutdown the launch file:
            plannerLaunch.shutdown()
            if timeOut:
                controllerLaunch.shutdown()

            # save data:
            if not timeOut:
                self.saveCollectedSamples()


            # reset doesn't clear the maxSamplesAchived flag
            self.reset()
            rospy.sleep(1)


            # for each 4 iterations (episods), save data
            if iteraction % 4 == 0 and self.store_data and self.dataWriter.CanAddSample():
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



