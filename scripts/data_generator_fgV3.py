# import sys
# ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
# if ros_path in sys.path:
#     sys.path.remove(ros_path)
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import signal
import sys
import threading
import math
import numpy as np
import pandas as pd
from numpy import linalg as la
import time
import datetime
import subprocess
import shutil
import pickle
from scipy.spatial.transform import Rotation
import rospy
import roslaunch
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, MultiArrayLayout, Empty as std_Empty
from geometry_msgs.msg import PoseStamped, Pose, Quaternion, Transform, Twist, TransformStamped, Vector3Stamped
from mav_planning_msgs.msg import PolynomialTrajectory4D, PolynomialSegment4D 
import tf
import tf2_geometry_msgs
from gazebo_msgs.msg import ModelState, LinkStates
from flightgoggles.msg import IRMarkerArray
from nav_msgs.msg import Path, Odometry
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
from sensor_msgs.msg import Image, Imu
from cv_bridge import CvBridge
# import cv2
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelState
from IrMarkersUtils import processMarkersMultiGate 
from store_read_data_extended import DataWriterExtended, DataReaderExtended
from Bezier_untils import bezier4thOrder, bezier2ndOrder, bezier3edOrder, bezier1stOrder
from environmentsCreation.FG_env_creator import readMarkrsLocationsFile
from environmentsCreation.gateNormalVector import computeGateNormalVector

SAVE_DATA_DIR = '/home/majd/catkin_ws/src/basic_rl_agent/data/stateAggregationDataFromTrackedTrajectories'

class StateAggregator:

    def __init__(self, camera_FPS=30, traj_length_per_image=30.9, dt=-1, numOfSamples=120, numOfDatapointsInFile=500, save_data_dir=None, twist_data_length=100):
        rospy.init_node('state_aggregator', anonymous=True)
        self.bridge = CvBridge()
        self.numOfDataPoints = numOfDatapointsInFile 
        self.camera_fps = camera_FPS
        self.traj_length_per_image = traj_length_per_image
        if dt == -1:
            self.numOfSamples = numOfSamples 
            self.dt = (self.traj_length_per_image/self.camera_fps)/self.numOfSamples
        else:
            self.dt = dt
            self.numOfSamples = (self.traj_length_per_image/self.camera_fps)/self.dt
        # pose variables
        self.tid_pose_dict = {}
        self.tid_orientation_dict = {}

        # twist storage variables
        self.twist_data_len = twist_data_length # we want twist_data_length with the same frequency of the odometry
        self.buff_maxSize = self.twist_data_len*50
        self.twist_tid_list = [] # stores the time as id from odometry msgs.
        self.tid_twist_dict = {} # stores the samples from odometry coming at ODOM_FREQUENCY.
        self.closestTidToTwistThreshold = 2 # it's the period of the odometry (1/500 = 0.002 [s]) represented in [ms]

        # acceleration variables
        self.acc_tid_list = []
        self.tid_acc_dict = {}

        self.trajectorySamplingPeriod = 0.01
        self.imageShape = (480, 640, 3) # (h, w, ch)

        ### markers/images variables
        self.markers_tid_list = []
        self.tid_images_dict = {}
        self.tid_markers_dict = {}
        self.irMarkersMsgCount = 0

        ### epoch handelling variables:
        self.gate6CenterWorld = np.array([0.0, 0.0, 2.038498]).reshape(3, 1)
        self.epoch_finished = True

        ### drone gate distance variables:
        self.droneGateDistanceLowerBound = 3 # observation
        self.droneGateDistanceUpperBound = 25 # guess

        self.stateAggregation_linearVelocityThreshold = 0.8 # observation
        self.stateAggregation_numOfImagesSequence = 4
        self.stateAggregation_numOfTwisSequence = 100
        self.stateAggregation_irMarkersSkipNum = 1
        self.stateAggregation_tidList = []
        self.tid_tsRosTime_dict = {}
        self.stateAggregation_sendCommandDict = {}
        self.stateAggregation_takeFirstXSamples = 10000
        self.stateAggregation_takenSamplesCount = 0

        ### ir_beacons variables
        self.targetGate = 'gate0B'
        markersLocationDir = '/home/majd/catkin_ws/src/basic_rl_agent/data/FG_linux/FG_gatesPlacementFile' 
        markersLocationDict = readMarkrsLocationsFile(markersLocationDir)
        targetGateMarkersLocation = markersLocationDict[self.targetGate]
        targetGateDiagonalLength = np.max([np.abs(targetGateMarkersLocation[0, :] - marker) for marker in targetGateMarkersLocation[1:, :]])
        # used for drone traversing check
        self.targetGateHalfSideLength = targetGateDiagonalLength/(2 * math.sqrt(2)) # [m]
        self.targetGateNormalVector, self.targetGateCOM = computeGateNormalVector(targetGateMarkersLocation)
        self.distanceFromTargetGateThreshold = 0.45 # found by observation # [m]
       
        ####################
        # dataWriter flags #
        ####################
        self.store_data = True # check SAVE_DATA_DIR

        # dataWriter stuff
        self.save_data_dir = save_data_dir
        if self.save_data_dir == None:
            self.save_data_dir = SAVE_DATA_DIR
        # create new directory for this run if store_data is True
        if self.store_data == True:
            self.save_data_dir = self.__createNewDirectory()
        self.dataWriter = self.__getNewDataWriter()


        ###### shutdown callback
        rospy.on_shutdown(self.shutdownCallback)
        ###### Subscribers:
        self.imu_subs = rospy.Subscriber('/hummingbird/ground_truth/imu', Imu, self.imuCallback, queue_size=100)
        self.odometry_subs = rospy.Subscriber('/hummingbird/ground_truth/odometry', Odometry, self.odometryCallback, queue_size=100)
        self.camera_subs = rospy.Subscriber('/uav/camera/left/image_rect_color', Image, self.rgbCameraCallback, queue_size=10)
        self.markers_subs = rospy.Subscriber('/uav/camera/left/ir_beacons', IRMarkerArray, self.irMarkersCallback, queue_size=10)
        # self.uav_collision_subs = rospy.Subscriber('/uav/collision', std_Empty, self.droneCollisionCallback, queue_size=1 )
        self.resetStateAggregation_subs = rospy.Subscriber('/state_aggregation/reset', std_Empty, self.stateAggregationResetCallback, queue_size=1)
        self.sampledTrajectoryChunk_subs = rospy.Subscriber('/hummingbird/sampledTrajectoryChunk', Float64MultiArray, self.sampleTrajectoryChunkCallback, queue_size=50)
        self.rvizPath_pub = rospy.Publisher('/state_aggregation_Path', Path, queue_size=1)

        ###### Publishers:
        self.sampleParticalTrajectory_pub = rospy.Publisher('/hummingbird/getTrajectoryChunk', Float64MultiArray, queue_size=1) 
        self.rvizPath_pub = rospy.Publisher('/path', Path, queue_size=1)
        self.dronePosePub = rospy.Publisher('/hummingbird/command/pose', PoseStamped, queue_size=1)

        # self.benchmarTimer = rospy.Timer(rospy.Duration(1/self.benchmarkCheckFreq), self.benchmarkTimerCallback, oneshot=False, reset=False)
        time.sleep(1)
    ############################################# end of init function

    def __createNewDirectory(self):
        dir_name = 'dataset_{}'.format(datetime.datetime.today().strftime('%Y%m%d%H%M_%S'))
        path = os.path.join(self.save_data_dir, dir_name)
        os.makedirs(path)
        return path

    def __getNewDataWriter(self):
        return DataWriterExtended(self.save_data_dir, self.dt, self.numOfSamples, self.numOfDataPoints, (self.stateAggregation_numOfImagesSequence, 1), (self.stateAggregation_numOfTwisSequence, 4), storeMarkers=True) # the shape of each vel data sample is (twist_data_len, 4) because we have velocity on x,y,z and yaw

    def shutdownCallback(self):
        if self.store_data:
            self.dataWriter.save_data()

    def imuCallback(self, msg):
        tid = int(msg.header.stamp.to_sec()*1000) 
        self.acc_tid_list.append(tid)
        accData = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
        self.tid_acc_dict[tid] = accData
        # self.acc_buff.append(accData)
        # check buff size
        if len(self.acc_tid_list) > self.buff_maxSize:
            self.acc_tid_list = self.acc_tid_list[-self.buff_maxSize :]     
        #     self.acc_buff = self.acc_buff[-self.buff_maxSize :]

    def odometryCallback(self, msg):
        self.lastOdomMsg = msg
        tid = int(msg.header.stamp.to_sec()*1000)
        twist = msg.twist.twist
        twist_data = np.array([twist.linear.x, twist.linear.y, twist.linear.z, twist.angular.z])
        # add pose data
        pose = msg.pose.pose
        q = pose.orientation
        curr_q = np.array([q.x, q.y, q.z, q.w])
        euler = Rotation.from_quat(curr_q).as_euler('xyz')
        pose_data = [pose.position.x, pose.position.y, pose.position.z, euler[2]]

        self.twist_tid_list.append(tid)
        self.tid_pose_dict[tid] = pose_data
        self.tid_twist_dict[tid] = twist_data
        self.tid_orientation_dict[tid] = curr_q
        # check buff size
        # if len(self.twist_tid_list) > self.buff_maxSize:
        #     self.twist_tid_list = self.twist_tid_list[-self.buff_maxSize :]     
        #     self.twist_buff = self.twist_buff[-self.buff_maxSize :]
        #     self.pose_buff = self.pose_buff[-self.buff_maxSize :]

    def irMarkersCallback(self, irMarkers_message):
        self.irMarkersMsgCount += 1
        if self.irMarkersMsgCount % self.stateAggregation_irMarkersSkipNum != 0:
            return
        gatesMarkersDict = processMarkersMultiGate(irMarkers_message)
        if self.targetGate in gatesMarkersDict.keys():
            markersData = gatesMarkersDict[self.targetGate]

            # check if all markers are visiable
            visiableMarkers = np.sum(markersData[:, -1] != 0)
            if  visiableMarkers <= 3:
                # print('not all markers are detected')
                return
            else: 
                # all markers are found
                # print('found {} markers'.format(visiableMarkers))
                tid = int(irMarkers_message.header.stamp.to_sec()*1000)
                self.stateAggregation_tidList.append(tid)
                self.markers_tid_list.append(tid)
                self.tid_markers_dict[tid] = markersData
                self.tid_tsRosTime_dict[tid] = irMarkers_message.header.stamp.to_sec()
        else:
            # print('no markers were found')
            pass

    def getImageMarkersDataTidSequence(self, tid, numOfImageSequence):
        curr_markers_tids = np.array(self.markers_tid_list)
        i = np.searchsorted(curr_markers_tids, tid, side='left')
        if (i != 0) and (curr_markers_tids[i] == tid) and (i >= numOfImageSequence-1):
            tid_sequence = curr_markers_tids[i-numOfImageSequence+1:i+1]
            # the tid diff is greater than 40ms, return None
            for k in range(numOfImageSequence-1):
                if tid_sequence[k+1] - tid_sequence[k] > 40:  
                    return None
            return tid_sequence

        return None
    
    def getTwistTidSequence(self, tid, numOfTwistSequence):
        curr_tid_nparray = np.array(self.twist_tid_list)
        idx = np.searchsorted(curr_tid_nparray, tid, side='left')
        if (idx < curr_tid_nparray.shape[0]-1) and (curr_tid_nparray[idx] == tid) and (idx < curr_tid_nparray.shape[0]-1) and (idx >= numOfTwistSequence-1):
            return curr_tid_nparray[idx-numOfTwistSequence+1:idx+1]
        return None
    
    def getClosestTidForOdomAndIMU(self, tid, tid_list, numOfTids):
        curr_tid_nparray = np.array(tid_list)
        idx = np.searchsorted(curr_tid_nparray, tid, side='left')
        if (idx < curr_tid_nparray.shape[0]-1):
            if abs(tid - curr_tid_nparray[idx]) <= self.closestTidToTwistThreshold:
                if (idx >= numOfTids-1):
                    return curr_tid_nparray[idx-numOfTids+1:idx+1]
                else:
                    return -1
            return tid - curr_tid_nparray[idx]
        return idx

    def rgbCameraCallback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        if cv_image.shape != self.imageShape:
            rospy.logwarn('the received image size is different from what expected')
        #     #cv_image = cv2.resize(cv_image, (self.imageShape[1], self.imageShape[0]))
        tid = int(msg.header.stamp.to_sec()*1000)
        self.tid_images_dict[tid] = cv_image

    def transformVector(self, vect, q):
        t = TransformStamped()
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        v = Vector3Stamped()
        v.vector.x = vect[0]
        v.vector.y = vect[1]
        v.vector.z = vect[2]

        vect_worldFrame = tf2_geometry_msgs.do_transform_vector3(v, t)
        return np.array([vect_worldFrame.vector.x, vect_worldFrame.vector.y, vect_worldFrame.vector.z])

    def checkTwistConditions(self, twist, orientaiton):
        v = twist[:-1] # remove yaw vel value
        linearVelocityMagnitued = la.norm(v)
        magnitudeCondition = linearVelocityMagnitued >= self.stateAggregation_linearVelocityThreshold
        v = v/linearVelocityMagnitued
        v = self.transformVector(v, orientaiton)
        innerProduct = np.inner(v, self.targetGateNormalVector) 
        directionCondition = (innerProduct < 0) and abs(innerProduct) > 0.25

        if not magnitudeCondition:
            print('too slow')
        elif not directionCondition:
            print('not going toward the gate. innerProduct={}'.format(innerProduct))

        return magnitudeCondition and directionCondition

    def stateAggregationResetCallback(self, msg):
        self.reset_variables()

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

    def sampleTrajectoryChunkCallback(self, msg):
        data = np.array(msg.data)
        msg_ts_rostime = data[0]
        print('new msg received from sampleTrajectoryChunkCallback msg_ts_rostime={} --------------'.format(msg_ts_rostime))
        data = data[1:]
        data_length = data.shape[0]
        assert data_length==4*self.numOfSamples, "Error in the received message"
        if self.store_data:
            if self.dataWriter.CanAddSample() == True:
                msg_tid = int(msg_ts_rostime*1000)
                Px, Py, Pz, Yaw = [], [], [], []
                for i in range(0, data.shape[0], 4):
                    # append the data to the variables:
                    Px.append(data[i])
                    Py.append(data[i+1])
                    Pz.append(data[i+2])
                    Yaw.append(data[i+3])

                # get stateAggregationDataDict
                stateAggregationDataDict = self.stateAggregation_sendCommandDict.pop(msg_tid, None)
                if stateAggregationDataDict is None:
                    rospy.WARN('stateAggregationDataDict for msg_tid: {} returned None'.format(msg_tid))
                    return

                # list of images to be saved
                imageList_sent = [stateAggregationDataDict['imageSeq']]
                markersDataList = np.array(stateAggregationDataDict['markersDataSeq'])
                twist_data_list = np.array(stateAggregationDataDict['twistDataSeq'])
                tid_sequence = stateAggregationDataDict['tid_sequence']

                # adding the sample
                self.dataWriter.addSample(Px, Py, Pz, Yaw, imageList_sent, tid_sequence, twist_data_list, markersDataList)

            else:
                if self.dataWriter.data_saved == False:
                    self.dataWriter.save_data()
                    rospy.logwarn('data saved.....')
                rospy.logwarn('cannot add samples, the maximum number of samples is reached.')
                self.dataWriter = self.__getNewDataWriter()
        try:
            self.publishSampledPathRViz(data, msg_ts_rostime)
        except:
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

    def generateRandomPose2(self, gateX, gateY, gateZ):
        xmin, xmax = -20, 20
        ymin, ymax = -40, -20
        zmin, zmax = gateZ - 1.0, gateZ + 2.0
        x = xmin + np.random.rand() * (xmax - xmin)
        y = ymin + np.random.rand() * (ymax - ymin)
        z = zmin + np.random.rand() * (zmax - zmin)
        minYaw, maxYaw = 0, 180
        yaw = minYaw + np.random.rand() * (maxYaw - minYaw)
        return x, y, z, yaw

    def reset_variables(self):
        # pose, twist, acc variables
        self.twist_tid_list = []
        self.tid_pose_dict = {}
        self.tid_twist_dict = {}
        self.acc_tid_list = []
        self.tid_acc_dict = {}

        # reset images/markers variabls
        self.markers_tid_list = []
        self.irMarkersMsgCount = 0
        self.tid_markers_dict = {}
        self.tid_images_dict = {}

        self.stateAggregation_tidList = []
        self.tid_tsRosTime_dict = {}



    def runOneEpoch(self): 
        self.epoch_finished = False
        rate = rospy.Rate(50)
        loopCount = 0
        while not self.epoch_finished and not rospy.is_shutdown():
            rate.sleep()
            loopCount += 1
            if loopCount > 22*50:
                self.epoch_finished = True
                print('timeout.')
                break

            if len(self.stateAggregation_tidList) < 3:
                continue

            tid = self.stateAggregation_tidList.pop(0)

            closestTwistTid = self.getClosestTidForOdomAndIMU(tid, self.twist_tid_list, 1)
            closestAccTid = self.getClosestTidForOdomAndIMU(tid, self.acc_tid_list, 1)
            if not isinstance(closestTwistTid, np.ndarray):
                if len(self.twist_tid_list) != 0:
                    print('closestTwistTid is', closestTwistTid, tid-self.twist_tid_list[-1])
                else:
                    print('self.twist_tid_list is empty')
                continue

            if not isinstance(closestAccTid, np.ndarray):
                if len(self.acc_tid_list) != 0:
                    print('closestAccTid is', closestAccTid, tid-self.acc_tid_list[-1])
                else:
                    print('self.acc_tid_list is empty')
                continue

            pose = self.tid_pose_dict.get(closestTwistTid[0], None)
            twist = self.tid_twist_dict.get(closestTwistTid[0], None)
            orientation = self.tid_orientation_dict.get(closestTwistTid[0], None)
            acc = self.tid_acc_dict.get(closestAccTid[0], None)

            if orientation is None:
                print('orientaiton is None')
                continue
            if twist is None:
                print('twist is None')
                continue
            if pose is None:
                print('pose is None')
                continue
            if acc is None:
                print('acc is None')
                continue
            
            # check drone pose related to the gate
            dronePosition = np.array(pose[:-1]) # remove the yaw value
            Vdg = dronePosition - self.targetGateCOM
            if la.norm(Vdg) < self.droneGateDistanceLowerBound:
                print('too close to the gate, Epoch finished')
                self.epoch_finished = True
                break
            elif la.norm(Vdg) > self.droneGateDistanceUpperBound:
                print('too far from the gate, skipping')
                continue

            if not self.checkTwistConditions(twist, orientation):
                continue
        
            twistTidSequence = self.getTwistTidSequence(closestTwistTid[0], self.stateAggregation_numOfTwisSequence)
            if twistTidSequence is None:
                print('twistTidSequence returned None')
                continue
            
            tid_sequence = self.getImageMarkersDataTidSequence(tid, self.stateAggregation_numOfImagesSequence)
            if tid_sequence is None:
                print('tid_seqeucne returned None')
                continue
            
            twistDataSeq = []
            for t in twistTidSequence:
                twistData = self.tid_twist_dict.get(t, None) 
                if twistData is None:
                    print('tid_twist_dict did not have value for t={}'.format(t))
                    continue
                twistDataSeq.append(twistData)

            imageSeq, markersDataSeq = [], []
            for t in tid_sequence:
                img = self.tid_images_dict.get(t, None)
                markersData = self.tid_markers_dict.get(t, None)
                if img is None or markersData is None:
                    print('img or markersData is None')
                    continue
                imageSeq.append(img)
                markersDataSeq.append(markersData)

            ts_rostime = self.tid_tsRosTime_dict.get(tid, None) 
            if ts_rostime is None:
                continue

            self.stateAggregation_sendCommandDict[tid] = {'imageSeq': imageSeq, 'markersDataSeq': markersDataSeq, 'tid_sequence':tid_sequence, 'twistDataSeq': twistDataSeq}
            self.sendCommand(ts_rostime)




    def run(self):
        gateX, gateY, gateZ = self.gate6CenterWorld.reshape(3, )
        for epoch in range(400):
            # Place the drone:
            droneX, droneY, droneZ, droneYaw = self.generateRandomPose2(gateX, gateY, gateZ)
            self.placeDrone(droneX, droneY, droneZ, droneYaw)
            self.pauseGazebo()
            time.sleep(0.8)
            self.pauseGazebo(False)

            self.reset_variables()
            time.sleep(0.5)

            plannerLaunch = self.__getPlanningLaunchObject()
            plannerLaunch.start()

            self.runOneEpoch()

            # shutdown the launch file:
            plannerLaunch.shutdown()

        # self.shutdownCallback()

def main():
    stateAgg = StateAggregator()
    stateAgg.run()


def signal_handler(sig, frame):
    sys.exit(0)   

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    main()