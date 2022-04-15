# import sys
# ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
# if ros_path in sys.path:
#     sys.path.remove(ros_path)
import sys
import os

from numpy.core.fromnumeric import std
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import signal
import sys
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
from std_msgs.msg import Empty as std_Empty
from geometry_msgs.msg import PoseStamped, Pose, Quaternion, Transform, Twist
# import tf
from gazebo_msgs.msg import ModelState, LinkStates
from flightgoggles.msg import IRMarkerArray
from nav_msgs.msg import Path, Odometry
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
from sensor_msgs.msg import Image
# import cv2
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelState
from IrMarkersUtils import processMarkersMultiGate 
from learning.MarkersToBezierRegression.markersToBezierRegressor_configurable import loadConfigsFromFile
from learning.MarkersToBezierRegression.markersToBezierRegressor_inferencing import MarkersAndTwistDataToBeizerInferencer
from Bezier_untils import bezier4thOrder, bezier2ndOrder, bezier3edOrder, bezier1stOrder
from environmentsCreation.FG_env_creator import readMarkrsLocationsFile
from environmentsCreation.gateNormalVector import computeGateNormalVector

class NetworkNavigatorBenchmarker:

    def __init__(self, networkConfig, weightsFile, camera_FPS=30, traj_length_per_image=30.9, dt=-1, numOfSamples=120, numOfDatapointsInFile=500, save_data_dir=None, twist_data_length=100):
        rospy.init_node('network_navigator_Benchmark', anonymous=True)
        # self.bridge = CvBridge()
        self.camera_fps = camera_FPS
        self.traj_length_per_image = traj_length_per_image
        if dt == -1:
            self.numOfSamples = numOfSamples 
            self.dt = (self.traj_length_per_image/self.camera_fps)/self.numOfSamples
        else:
            self.dt = dt
            self.numOfSamples = (self.traj_length_per_image/self.camera_fps)/self.dt


        # twist storage variables
        self.twist_data_len = twist_data_length # we want twist_data_length with the same frequency of the odometry
        self.twist_buff_maxSize = self.twist_data_len*50
        self.twist_tid_list = [] # stores the time as id from odometry msgs.
        self.twist_buff = [] # stores the samples from odometry coming at ODOM_FREQUENCY.


        self.trajectorySamplingPeriod = 0.01
        self.curr_sample_time = rospy.Time.now().to_sec()
        self.curr_trajectory = None
        self.T = self.numOfSamples * self.dt
        acc = 30
        self.t_space = np.linspace(0, 1, acc) #np.linspace(0, self.numOfSamples*self.dt, self.numOfSamples)
        self.imageShape = (480, 640, 3) # (h, w, ch)
        
        self.t_id = 0
        self.networkInferencer = MarkersAndTwistDataToBeizerInferencer(self.imageShape, networkConfig, weightsFile)
        self.networkConfig = networkConfig
        self.numOfImageSequence = self.networkConfig.get('numOfImageSequence', 1)
        self.numOfTwistSequence = self.networkConfig.get('numOfTwistSequence', 100)

        self.lastIrMarkersMsgTime = None
        self.IrMarkersMsgIntervalSum = 0
        self.IrMarerksMsgCount_FPS = 0
        self.irMarkersMsgCount = 0
        self.noMarkersFoundCount = 0
        self.noMarkersFoundThreshold = 90 # 3 secs for 30 FPS

        self.expected_markers_time_diff = 16 # the expected time diff between two frames, depends of the data collection
        self.markers_tid_list = []
        self.tid_markers_dict = {}

        ##########################
        ## benchmark variables: ##
        ##########################
        self.save_benchmark_results = True
        self.benchmarkSaveResultsDir = '/home/majd/catkin_ws/src/basic_rl_agent/data/deep_learning/benchmarks/results'
        assert os.path.exists(self.benchmarkSaveResultsDir), 'self.benchmarkSaveResultsDir does not exist'

        startIndex = weightsFile.rfind('config{}'.format(self.networkConfig['configNum']), 0)
        assert startIndex != -1, 'configNum was not found in the provided network weightsFile.'
        self.benchmark_find_name = weightsFile[startIndex:].split('.')[0]

        self.benchmarkCheckFreq = 30
        self.TIMEOUT_SEC = 20 # [sec] 
        self.roundTimeOutCount = self.TIMEOUT_SEC * self.benchmarkCheckFreq # [sec/sec]
        self.benchmarking = False
        self.benchamrkPoseDataBuffer = []
        self.benchmarkTwistDataBuffer = []
        self.benchmarkTimerCount = 0
        self.roundFinishReason = 'unknow'
        self.benchmarkResultsDict = {
            'pose': [],
            'round_finish_reason': [],
            'average_twist': [],
            'peak_twist': [],
            'average_FPS': [],
            'traverseDistanceFromTheCenterOfTheGate': [],
            'distanceFromDronesPositionToTargetGateCOM': []
        }

        # ir_beacons variables
        self.targetGate = 'gate0B'
        markersLocationDir = '/home/majd/catkin_ws/src/basic_rl_agent/data/FG_linux/FG_gatesPlacementFileV2' 
        markersLocationDict = readMarkrsLocationsFile(markersLocationDir)
        targetGateMarkersLocation = markersLocationDict[self.targetGate]
        targetGateDiagonalLength = np.max([np.abs(targetGateMarkersLocation[0, :] - marker) for marker in targetGateMarkersLocation[1:, :]])
        # used for drone traversing check
        self.targetGateHalfSideLength = targetGateDiagonalLength/(2 * math.sqrt(2)) * 1.1 # [m]
        self.targetGateNormalVector, self.targetGateCOM = computeGateNormalVector(targetGateMarkersLocation)
        self.distanceFromTargetGateThreshold = 0.45 # found by observation # [m]
        self.lastVdg = None
        self.traverseDistanceFromTheCenterOfTheGate = 1000000
        self.distanceFromDronesPositionToTargetGateCOM = 1000000
       
        # Subscribers:
        self.odometry_subs = rospy.Subscriber('/hummingbird/ground_truth/odometry', Odometry, self.odometryCallback, queue_size=1)
        self.camera_subs = rospy.Subscriber('/uav/camera/left/image_rect_color', Image, self.rgbCameraCallback, queue_size=1)
        self.markers_subs = rospy.Subscriber('/uav/camera/left/ir_beacons', IRMarkerArray, self.irMarkersCallback, queue_size=1)
        self.uav_collision_subs = rospy.Subscriber('/uav/collision', std_Empty, self.droneCollisionCallback, queue_size=1 )

        # Publishers:
        self.trajectory_pub = rospy.Publisher('/hummingbird/command/trajectory', MultiDOFJointTrajectory,queue_size=1)
        self.dronePosePub = rospy.Publisher('/hummingbird/command/pose', PoseStamped, queue_size=1)
        self.rvizPath_pub = rospy.Publisher('/path', Path, queue_size=1)
        # sent to State Aggregation node if available.
        self.resetStateAggregation_pub = rospy.Publisher('/state_aggregation/reset', std_Empty, queue_size=1)

        self.trajectorySamplingTimer = rospy.Timer(rospy.Duration(self.trajectorySamplingPeriod), self.timerCallback, oneshot=False, reset=False)
        self.benchmarTimer = rospy.Timer(rospy.Duration(1/self.benchmarkCheckFreq), self.benchmarkTimerCallback, oneshot=False, reset=False)
        time.sleep(1)
    ############################################# end of init function

    def odometryCallback(self, msg):
        self.lastOdomMsg = msg
        t_id = int(msg.header.stamp.to_sec()*1000)
        pose = msg.pose.pose
        twist = msg.twist.twist
        twist_data = np.array([twist.linear.x, twist.linear.y, twist.linear.z, twist.angular.z])
        self.twist_tid_list.append(t_id)
        self.twist_buff.append(twist_data)
        if len(self.twist_buff) > self.twist_buff_maxSize:
            self.twist_buff = self.twist_buff[-self.twist_buff_maxSize :]
            self.twist_tid_list = self.twist_tid_list[-self.twist_buff_maxSize :]     
        if self.benchmarking:
            self.benchmarkTwistDataBuffer.append(twist_data)
            # quaternion = ( pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
            # yaw = tf.transformations.euler_from_quaternion(quaternion)[2]
            # poseArray = np.array([pose.position.x, pose.position.y, pose.position.z, yaw])
            # self.benchamrkPoseDataBuffer.append(poseArray)

    def droneCollisionCallback(self, msg):
        if self.benchmarking:
            print('drone collided! round finished')
            self.roundFinishReason = 'droneCollided'
            self.roundFinished = True
            self.benchmarking = False

    def publishSampledPathRViz(self, positionCP):
        now = rospy.Time.now()
        now_secs = now.to_sec()
        poses_list = []
        for ti in self.t_space:
            Pxyz = bezier4thOrder(positionCP, ti)
            poseStamped_msg = PoseStamped()    
            poseStamped_msg.header.stamp = rospy.Time.from_sec(now_secs + ti*4)
            poseStamped_msg.header.frame_id = 'world'
            poseStamped_msg.pose.position.x = Pxyz[0]
            poseStamped_msg.pose.position.y = Pxyz[1]
            poseStamped_msg.pose.position.z = Pxyz[2]
            quat = [0, 0, 0, 1] #tf.transformations.quaternion_from_euler(0, 0, data[i+3])
            poseStamped_msg.pose.orientation.x = quat[0]
            poseStamped_msg.pose.orientation.y = quat[1]
            poseStamped_msg.pose.orientation.z = quat[2]
            poseStamped_msg.pose.orientation.w = quat[3]
            poses_list.append(poseStamped_msg)
        path = Path()
        path.poses = poses_list        
        path.header.stamp = now
        path.header.frame_id = 'world'
        self.rvizPath_pub.publish(path)

    def timerCallback(self, timerMsg):
        if self.curr_trajectory is None:
            return
        positionCP = self.curr_trajectory[0]
        yawCP = self.curr_trajectory[1]
        currTime = self.curr_trajectory[2]

        t = (rospy.Time.now().to_sec() - currTime)/self.customT
        if t > 1:
            return
        Pxyz = bezier4thOrder(positionCP, t)
        Pyaw = bezier2ndOrder(yawCP, t) 
        q = Rotation.from_euler('z', Pyaw).as_quat()[0]
        transform = Transform()
        transform.translation.x = Pxyz[0]
        transform.translation.y = Pxyz[1]
        transform.translation.z = Pxyz[2]
        transform.rotation.x = q[0] 
        transform.rotation.y = q[1] 
        transform.rotation.z = q[2] 
        transform.rotation.w = q[3] 

        # computing velocities:
        linearVelCP = np.zeros((3, 4))
        for i in range(4):
            linearVelCP[:, i] = positionCP[:, i+1]-positionCP[:, i]
        linearVelCP = 4 * linearVelCP
        Vxyz = bezier3edOrder(linearVelCP, t)

        angularVelCP = np.zeros((1, 2))
        for i in range(2):
            angularVelCP[:, i] = yawCP[:, i+1] - yawCP[:, i]
        angularVelCP = 2 * angularVelCP
        Vyaw = bezier1stOrder(angularVelCP, t) 

        vel_twist = Twist()
        vel_twist.linear.x = Vxyz[0]
        vel_twist.linear.y = Vxyz[1]
        vel_twist.linear.z = Vxyz[2]
        vel_twist.angular.x = 0
        vel_twist.angular.y = 0
        vel_twist.angular.z = Vyaw

        # compute accelerations:
        linearAccCP = np.zeros((3, 3))
        for i in range(3):
            linearAccCP[:, i] = linearVelCP[:, i+1]-linearVelCP[:, i]
        linearAccCP = 3 * linearAccCP
        Axyz = bezier2ndOrder(linearAccCP, t)

        # the angular accelration is constant since the yaw is second order polynomial
        angularAcc = angularVelCP[0, 1] - angularVelCP[0, 0]

        Acc_twist = Twist()
        Acc_twist.linear.x = Axyz[0]
        Acc_twist.linear.y = Axyz[1]
        Acc_twist.linear.z = Axyz[2]
        Acc_twist.angular.x = 0
        Acc_twist.angular.y = 0
        Acc_twist.angular.z = angularAcc

        point = MultiDOFJointTrajectoryPoint()
        point.transforms = [transform]
        point.velocities = [vel_twist]
        point.accelerations = [Acc_twist]

        point.time_from_start = rospy.Duration(self.curr_sample_time)
        self.curr_sample_time += self.trajectorySamplingPeriod

        trajectory = MultiDOFJointTrajectory()
        trajectory.points = [point]
        trajectory.header.stamp = rospy.Time.now()
        trajectory.joint_names = ['base_link']
        try:
            self.trajectory_pub.publish(trajectory)
        except:
            pass
        
    def processControlPoints(self, positionCP, yawCP, currTime):
        odom = self.lastOdomMsg
        q = odom.pose.pose.orientation
        curr_q = np.array([q.x, q.y, q.z, q.w])
        euler = Rotation.from_quat(curr_q).as_euler('xyz')
        currYaw = euler[-1]
        rotMat = Rotation.from_euler('z', currYaw).as_dcm()
        positionCP = np.matmul(rotMat, positionCP)
        trans_world = odom.pose.pose.position
        trans_world = np.array([trans_world.x, trans_world.y, trans_world.z]).reshape(3, 1)
        positionCP_world = positionCP + trans_world 
        # add current yaw to the yaw control points
        yawCP = yawCP + currYaw
        self.curr_trajectory = [positionCP_world, yawCP, currTime.to_sec()]
        try:
            self.publishSampledPathRViz(positionCP_world)
        except:
            pass

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

    def irMarkersCallback(self, irMarkers_message):
        if not self.benchmarking:
            return

        self.irMarkersMsgCount += 1
        if self.irMarkersMsgCount % 1 != 0:
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
                # print('found {} markers'.format(visiableMarkers))
                self.currTime = rospy.Time.now()
                t_id = int(irMarkers_message.header.stamp.to_sec()*1000)
                self.markers_tid_list.append(t_id)
                self.tid_markers_dict[t_id] = markersData
                self.t_id = t_id

            self.noMarkersFoundCount = 0
        else:
            # print('no markers were found')
            self.noMarkersFoundCount += 1
            self.lastIrMarkersMsgTime = None

        if self.lastIrMarkersMsgTime is None:
            self.lastIrMarkersMsgTime = irMarkers_message.header.stamp.to_sec()
            return
        msgTime = irMarkers_message.header.stamp.to_sec() 
        self.IrMarkersMsgIntervalSum += msgTime - self.lastIrMarkersMsgTime
        self.IrMarerksMsgCount_FPS += 1
        self.lastIrMarkersMsgTime = msgTime
        # print('average FPS = ', self.IrMarerksMsgCount_FPS/self.IrMarkersMsgIntervalSum)

    def getMarkersDataSequence(self, tid):
        curr_markers_tids = np.array(self.markers_tid_list)

        i = np.searchsorted(curr_markers_tids, tid, side='left')

        if (i != 0) and (i < curr_markers_tids.shape[0]) and (curr_markers_tids[i] == tid) and (i >= self.numOfImageSequence-1):
            tid_sequence = curr_markers_tids[i-self.numOfImageSequence+1:i+1]

            # if tid diff is greater/smaller than (expected_markers_time_diff) by 10 ms, raise warning and return None
            for k in range(self.numOfImageSequence-1):
                diff = tid_sequence[k+1] - tid_sequence[k]
                print(diff)
                if abs(diff - self.expected_markers_time_diff) > 10:
                    rospy.logwarn('the time diff {} between markers frames is not close to the expected one {}, returning None'.format(diff, self.expected_markers_time_diff))
                    return None
            return tid_sequence
        
        if i >= curr_markers_tids.shape[0]:
            print('tid was not found')
            print(curr_markers_tids.shape)
            print(tid, curr_markers_tids[-10:]) 

        return None

    def rgbCameraCallback(self, image_message):
        pass
        # cv_image = self.bridge.imgmsg_to_cv2(image_message, desired_encoding='bgr8')
        # if cv_image.shape != self.imageShape:
        #     rospy.logwarn('the received image size is different from what expected')
        #     #cv_image = cv2.resize(cv_image, (self.imageShape[1], self.imageShape[0]))
        # ts_rostime = image_message.header.stamp.to_sec()
    
    def placeDrone(self, x, y, z, yaw=-1, qx=0, qy=0, qz=0, qw=0):
        # if yaw is provided (in degrees), then caculate the quaternion
        if yaw != -1:
            q = Rotation.from_euler('z', yaw*math.pi/180.0).as_quat()
            # q = tf.transformations.quaternion_from_euler(0, 0, yaw*math.pi/180.0) 
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

    def generateRandomPose(self, gateX, gateY, gateZ, maxYawRotation=60):
        xmin, xmax = gateX - 3, gateX + 3
        ymin, ymax = gateY - 16, gateY - 25
        zmin, zmax = gateZ - 0.8, gateZ + 2.0
        x = xmin + np.random.rand() * (xmax - xmin)
        y = ymin + np.random.rand() * (ymax - ymin)
        z = zmin + np.random.rand() * (zmax - zmin)
        yaw = np.random.normal(90, maxYawRotation/5) # 99.9% of the samples are in 5*segma
        # if np.random.rand() > 0.5:
        #     yawMin, yawMax = 60, 70
        # else:
        #     yawMin, yawMax = 110, 120
        # yaw = yawMin + np.random.rand() * (yawMax-yawMin)
        return x, y, z, yaw

    def benchmarkTimerCallback(self, timerMsg):
        if not self.benchmarking:
            return 

        # check if the drone traversed the gate:
        position = self.lastOdomMsg.pose.pose.position
        dronePosition = np.array([position.x, position.y, position.z])
        Vdg = dronePosition - self.targetGateCOM
        if self.lastVdg is None:
            self.lastVdg = Vdg
            return

        # compute the dot products: The abs of the dot product is the distance between the drone and the gate plane 
        curr_d1 = np.inner(self.targetGateNormalVector, Vdg) 
        last_d1 = np.inner(self.targetGateNormalVector, self.lastVdg)  # between -1 and 1

        # the distance between the drone and the gate COM
        curr_d3 = la.norm(Vdg)
        last_d3 = la.norm(self.lastVdg)

        # the distance of the projected position of the drone to the gate's plane and the gate COM
        curr_d2 = math.sqrt(curr_d3**2 - curr_d1**2)
        last_d2 = math.sqrt(last_d3**2 - last_d1**2)
        if curr_d1 < 0 and curr_d2 < self.targetGateHalfSideLength and \
            last_d1 > 0 and  last_d2 < self.targetGateHalfSideLength:
            # print(curr_d1, last_d1, curr_d2, last_d2)
            self.roundFinishReason = 'dronePassedGate'
            self.benchmarkTimerCount = 0
            self.roundFinished = True
            self.benchmarking = False
            self.traverseDistanceFromTheCenterOfTheGate = (curr_d2 + last_d2)/2
            self.distanceFromDronesPositionToTargetGateCOM = curr_d3
            print('drone passed the gate!')

        # save Vdg for the next step
        self.lastVdg = Vdg

        # check if round finished or no markers found count threshold
        self.benchmarkTimerCount += 1
        if self.benchmarkTimerCount >= self.roundTimeOutCount or \
                    self.noMarkersFoundCount > self.noMarkersFoundThreshold:

            # check if the drone is in front of the gate
            self.traverseDistanceFromTheCenterOfTheGate = curr_d2
            self.distanceFromDronesPositionToTargetGateCOM = curr_d3
            if curr_d1 < self.distanceFromTargetGateThreshold and curr_d2 < self.targetGateHalfSideLength:
                # print(self.distanceFromDronesPositionToTargetGateCOM, self.traverseDistanceFromTheCenterOfTheGate)
                print('droneInFrontOfGate. round finished')
                self.roundFinishReason = 'droneInFrontOfGate'
                self.traverseDistanceFromTheCenterOfTheGate = curr_d2
                self.distanceFromDronesPositionToTargetGateCOM = curr_d3
                self.benchmarkTimerCount = 0
                self.roundFinished = True
                self.benchmarking = False
            # no, then timeout!
            else:
                if self.benchmarkTimerCount >= self.roundTimeOutCount:
                    print('timeout! round finished.')
                    self.roundFinishReason = 'timeOut'
                else:
                    print('noMarkersFoundThreshold reached. round finished.')
                    self.roundFinishReason = 'noMarkersFoundThreshold'
                self.roundFinished = True
                self.benchmarking = False

    def reset_variables(self):
        # reset state aggregation node, if availabe
        self.resetStateAggregation_pub.publish(std_Empty())

        self.lastVdg = None
        self.roundFinishReason = 'unknown'
        self.noMarkersFoundCount = 0
        self.IrMarkersMsgIntervalSum = 0
        self.lastIrMarkersMsgTime = None
        self.IrMarerksMsgCount_FPS = 0
        self.benchamrkPoseDataBuffer = []
        self.benchmarkTwistDataBuffer = [] 
        self.traverseDistanceFromTheCenterOfTheGate = 1000000
        self.distanceFromDronesPositionToTargetGateCOM = 1000000

        self.markers_tid_list = []
        self.tid_markers_dict = {}

        self.roundFinished = False
        self.benchmarkTimerCount = 0


    def run(self, PosesfileName, poses):
        '''
            @param poese: a list of np arraies. each np array has an initial pose (x, y, z, yaw).

            each pose with a target_FPS correspond to a round.
            The round is finished if the drone reached the gate or if the roundTimeOut accured or if the drone is collided.
        '''

        self.customT = self.T * 1

        inference_time_list = []
        for roundId, pose in enumerate(poses):
            print('\nconfig{}, processing round {}:'.format(self.networkConfig['configNum'], roundId), end=' ')
            # Place the drone:
            droneX, droneY, droneZ, droneYaw = pose
            self.curr_trajectory = None

            self.placeDrone(droneX, droneY, droneZ, droneYaw)
            self.pauseGazebo()
            time.sleep(0.8)
            self.pauseGazebo(False)
            time.sleep(0.8)

            # variables preparation for a new round:
            self.reset_variables()

            # start the benchmarking
            self.benchmarking = True

            counter = 0
            self.frameMode = 15

            while not rospy.is_shutdown() and not self.roundFinished:

                # check if there are new markers data
                if self.t_id != 0:
                    # save current time
                    currTime = self.currTime

                    # markersData preprocessing 
                    tid_sequence = self.getMarkersDataSequence(self.t_id)
                    if tid_sequence is None:
                        # rospy.logwarn('tid_sequence returned None')
                        self.t_id = 0
                        continue
                    markersDataSeq = []
                    for tid in tid_sequence:
                        markersDataSeq.append(self.tid_markers_dict[tid]) 
                    currMarkersData = np.array(markersDataSeq)

                    self.t_id = 0

                    # twist data preprocessing
                    if len(self.twist_buff) < self.numOfTwistSequence:
                        continue
                    currTwistData = np.array(self.twist_buff[-self.numOfTwistSequence:])

                    ts = time.time()
                    y_hat = self.networkInferencer.old_normalizing_inference(currMarkersData, currTwistData)
                    inference_time = time.time() - ts

                    positionCP, yawCP = y_hat[0][0].numpy(), y_hat[1][0].numpy()
                    positionCP = positionCP.reshape(5, 3).T
                    yawCP = yawCP.reshape(1, 3)

                    if counter % self.frameMode== 0:
                        self.processControlPoints(positionCP, yawCP, currTime)
                    counter += 1

                    inference_time_list.append(inference_time)
                    mean_inference_time = np.array(inference_time_list).mean()
                    print('inference time: mean: {}, Hz: {}'.format(mean_inference_time, 1.0/mean_inference_time))

                    # self.networkInferencer.reset_states()
    
            # process benchmark data:
            if self.roundFinished:
                self.benchmarking = False
                self.processBenchmarkingData(pose)

        # end of the for loop
        mean_inference_time = np.array(inference_time_list).mean()
        print('inference time: mean: {}, Hz: {}'.format(mean_inference_time, 1.0/mean_inference_time))

        # saving the results
        if self.save_benchmark_results:
            benchmark_fileName_with_posesFileName = '{}_{}_frameMode{}_{}.pkl'.format(self.benchmark_find_name, PosesfileName.split('.')[0], self.frameMode, datetime.datetime.today().strftime('%Y%m%d%H%M_%S'))
            with open(os.path.join(self.benchmarkSaveResultsDir, benchmark_fileName_with_posesFileName), 'wb') as file_out:
                pickle.dump(self.benchmarkResultsDict, file_out) 
            print('{} was saved!'.format(benchmark_fileName_with_posesFileName))

    ################################### end of run function 

    def processBenchmarkingData(self, pose):
        # peak and average speed:
        self.benchmarkTwistDataBuffer = np.array(self.benchmarkTwistDataBuffer)
        try:
            averageTwist = np.mean(self.benchmarkTwistDataBuffer, axis=0)
            print(self.benchmarkTwistDataBuffer.shape, averageTwist.shape)
            linearVel = self.benchmarkTwistDataBuffer[:, :-1] # remove the angular yaw velocity
            linearVel = la.norm(linearVel, axis=1) 
            peakTwist = np.max(linearVel)
        except Exception as e:
            print(e)
            print('skipping')
            averageTwist = None
            linearVel = None
            peakTwist = None

        self.benchmarkResultsDict['pose'].append(pose)
        self.benchmarkResultsDict['round_finish_reason'].append(self.roundFinishReason)
        self.benchmarkResultsDict['average_twist'].append(averageTwist)
        self.benchmarkResultsDict['peak_twist'].append(peakTwist)
        self.benchmarkResultsDict['traverseDistanceFromTheCenterOfTheGate'].append(self.traverseDistanceFromTheCenterOfTheGate)
        self.benchmarkResultsDict['distanceFromDronesPositionToTargetGateCOM'].append(self.distanceFromDronesPositionToTargetGateCOM)
        if self.IrMarerksMsgCount_FPS != 0:
            self.benchmarkResultsDict['average_FPS'].append(self.IrMarerksMsgCount_FPS/self.IrMarkersMsgIntervalSum)
        else:
            self.benchmarkResultsDict['average_FPS'].append(-1)

    def benchmark(self, benchmarkPosesRootDir, fileName):
        posesDataFrame = pd.read_pickle(os.path.join(benchmarkPosesRootDir, fileName))
        poses = posesDataFrame['poses'].tolist()
        self.run(fileName, poses)

    def generateBenchmarkPosesFile(self, fileName, numOfPoses):
        gateX, gateY, gateZ = self.targetGateCOM.reshape(3, )
        posesList = []
        for i in range(numOfPoses):
            pose = self.generateRandomPose(gateX, gateY, gateZ, maxYawRotation=35)
            posesList.append(np.array(pose))
        df = pd.DataFrame({
            'poses': posesList
        })
        df.to_pickle(fileName)
        print('generated {} with {} poses'.format(fileName, numOfPoses))

def signal_handler(sig, frame):
    sys.exit(0)   

def generateBenchhmarkerPosesFile(numOfPoses):
    configs_file = '/home/majd/catkin_ws/src/basic_rl_agent/scripts/learning/MarkersToBezierRegression/configs/configs1.yaml'
    configs = loadConfigsFromFile(configs_file)
    weightsFile = '/home/majd/catkin_ws/src/basic_rl_agent/data/deep_learning/MarkersToBezierDataFolder/models_weights/weights_MarkersToBeizer_FC_scratch_withYawAndTwistData_config19_20210827-060041.h5'
    benchmarkPosesRootDir = '/home/majd/catkin_ws/src/basic_rl_agent/data/deep_learning/benchmarks/benchmarkPosesFiles'
    benchmarkerPosesFile = 'benchmarkerPosesFile_#{}_{}.pkl'.format(numOfPoses, datetime.datetime.today().strftime('%Y%m%d%H%M_%S'))
    networkBenchmarker = NetworkNavigatorBenchmarker(networkConfig=configs['config19'], weightsFile=weightsFile)
    networkBenchmarker.generateBenchmarkPosesFile(fileName=os.path.join(benchmarkPosesRootDir, benchmarkerPosesFile), numOfPoses=numOfPoses) 
    exit()

def loadConfigFiles(listOfConfigNums=None):
    configFiles = []
    configDir = '/home/majd/catkin_ws/src/basic_rl_agent/scripts/learning/MarkersToBezierRegression/configs'
    for configFile in [file for file in os.listdir(configDir) if file.endswith('.yaml')]:
        configFiles.append(os.path.join(configDir, configFile))

    allConfigs = {}
    for file in configFiles:
        configs = loadConfigsFromFile(file)
        # update the missing configs
        for config in configs.keys():
           configs[config]['numOfImageSequence'] = configs[config].get('numOfImageSequence', 1)
           configs[config]['markersNetworkType'] = configs[config].get('markersNetworkType', 'Dense')
           configs[config]['twistNetworkType'] = configs[config].get('twistNetworkType', 'Dense')
           configs[config]['twistDataGenType'] = configs[config].get('twistDataGenType', 'last2points')
        if listOfConfigNums is None:
            allConfigs.update(configs)
        else:
            tmpConfigs = {}
            for key, config in configs.items():
                if 'config{}'.format(config['configNum']) in listOfConfigNums:
                    tmpConfigs[key] = config
            if tmpConfigs:
                allConfigs.update(tmpConfigs)
    return allConfigs
        
def loadWeightsForConfigs(skipExistedFiles=False, listOfConfigNums=None):
    benchmarkSaveResultsDir = '/home/majd/catkin_ws/src/basic_rl_agent/data/deep_learning/benchmarks/results'
    existedFiles = os.listdir(benchmarkSaveResultsDir)
    
    weightsDir = '/home/majd/catkin_ws/src/basic_rl_agent/data/deep_learning/MarkersToBezierDataFolder/models_weights_for_benchmark'
    allWeights = os.listdir(weightsDir)

    allConfigs = loadConfigFiles(listOfConfigNums)

    configWeightTupleList = []
    for key in allConfigs.keys():
        for weight in allWeights:
            if key in weight:
                if skipExistedFiles:
                    l1 = [file for file in existedFiles if key in file and weight in file]
                    if len(l1) != 0:
                        print('skipping files: {}', l1)
                        continue
                configWeightTupleList.append((allConfigs[key], os.path.join(weightsDir, weight) ) )

    # print(configWeightTupleList)
    return configWeightTupleList

def benchmarkAllConfigsAndWeights(skipExistedFiles, listOfConfigNums=None):
    benchmarkPosesRootDir = '/home/majd/catkin_ws/src/basic_rl_agent/data/deep_learning/benchmarks/benchmarkPosesFiles'
    posesFiles = os.listdir(benchmarkPosesRootDir)

    configWeightTupleList = loadWeightsForConfigs(skipExistedFiles, listOfConfigNums)
    for config, weight in configWeightTupleList:
        for fileName in posesFiles:
            if 'ignore' in fileName:
                continue
            # try:
            print('############################################')
            print('processing file: config{}, weights: {}'.format(config['configNum'], weight.split('/')[-1] ) )
            networkBenchmarker = NetworkNavigatorBenchmarker(networkConfig=config, weightsFile=weight)
            networkBenchmarker.benchmark(benchmarkPosesRootDir, fileName)
            # except rospy.ROSInterruptException as e:
            #     print('rospy excption catched')
            #     print(e)
            #     exit()
            # except Exception as e:
            #     print(e)

def benchmarkSigleConfigNum(configNum, weight):
    benchmarkPosesRootDir = '/home/majd/catkin_ws/src/basic_rl_agent/data/deep_learning/benchmarks/benchmarkPosesFiles'
    posesFiles = os.listdir(benchmarkPosesRootDir)


    allConfigs = loadConfigFiles() 
    config = allConfigs[configNum]

    print(config)
    for fileName in posesFiles:
        if 'ignore' in fileName:
            continue
        print('############################################')
        print('processing file: config{}, weights: {}'.format(config['configNum'], weight.split('/')[-1] ) )
        networkBenchmarker = NetworkNavigatorBenchmarker(networkConfig=config, weightsFile=weight)
        networkBenchmarker.benchmark(benchmarkPosesRootDir, fileName)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    # generateBenchhmarkerPosesFile(100) # check random_pose_generation settings

    # listOfConfigNums = ['config15', 'config16', 'config17', 'config20', 'config26']
    # listOfConfigNums = ['config61'] #'config37', 'config35'] #, 'config30']
    # benchmarkAllConfigsAndWeights(skipExistedFiles=True, listOfConfigNums=listOfConfigNums)

    checkpoint_path = '/home/majd/catkin_ws/src/basic_rl_agent/data/deep_learning/MarkersToBezierDataFolder/models_weights/weights_MarkersToBeizer_FC_scratch_withYawAndTwistData_config37_20210829-134729.h5'
    benchmarkSigleConfigNum('config37', checkpoint_path)
    
    



