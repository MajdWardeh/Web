# import sys
# ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
# if ros_path in sys.path:
#     sys.path.remove(ros_path)
import sys
import os
from turtle import position


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

os.sys.path.append('/home/majd/catkin_ws/src/basic_rl_agent/scripts')
os.sys.path.append('/home/majd/papers/Python-B-spline-examples')

from IrMarkersUtils import processMarkersMultiGate 
# from learning.MarkersToBezierRegression.markersToBezierRegressor_configurable_withCostumLoss import loadConfigsFromFile
# from learning.MarkersToBezierRegression.markersToBezierRegressor_inferencing import MarkersAndTwistDataToBeizerInferencer
# from Bezier_untils import bezier4thOrder, bezier2ndOrder, bezier3edOrder, bezier1stOrder
from environmentsCreation.FG_env_creator import readMarkrsLocationsFile
from environmentsCreation.gateNormalVector import computeGateNormalVector
from scipy_minimize_bspline_3D_with_orientation_motionBlur import solve_n_D_OptimizationProblem

class TrajectoryNavigator:

    def __init__(self, twist_data_length=20):
        rospy.init_node('trajectory_navigator', anonymous=True)
        # self.camera_fps = camera_FPS
        # self.traj_length_per_image = traj_length_per_image
        # if dt == -1:
        #     self.numOfSamples = numOfSamples 
        #     self.dt = (self.traj_length_per_image/self.camera_fps)/self.numOfSamples
        # else:
        #     self.dt = dt
        #     self.numOfSamples = (self.traj_length_per_image/self.camera_fps)/self.dt


        # twist storage variables
        self.twist_data_len = twist_data_length # we want twist_data_length with the same frequency of the odometry
        self.twist_buff_maxSize = self.twist_data_len*50
        self.twist_tid_list = [] # stores the time as id from odometry msgs.
        self.twist_buff = [] # stores the samples from odometry coming at ODOM_FREQUENCY.


        self.trajectorySamplingPeriod = 0.02
        self.curr_sample_time = rospy.Time.now().to_sec()
        self.ti_index = -1
        acc = 30
        self.t_space = np.linspace(0, 1, acc) #np.linspace(0, self.numOfSamples*self.dt, self.numOfSamples)
        self.imageShape = (480, 640, 3) # (h, w, ch)
        
        self.gateMarkers = None

        self.lastIrMarkersMsgTime = None
        self.IrMarkersMsgIntervalSum = 0
        self.IrMarerksMsgCount_FPS = 0
        self.irMarkersMsgCount = 0
        self.noMarkersFoundCount = 0
        self.noMarkersFoundThreshold = 90 # 3 secs for 30 FPS

        self.markers_tid_list = []
        self.tid_markers_dict = {}


        ### Gates variables:
        self.targetGateList = ['gate0B'] #, 'gate1B', 'gate2B']
        self.targetGateIndex = 0

        markersLocationDir = '/home/majd/catkin_ws/src/basic_rl_agent/data/FG_linux/FG_gatesPlacementFileV2' 
        self.markersLocationDict = readMarkrsLocationsFile(markersLocationDir)

        self.computeGateVariables(self.targetGateList[self.targetGateIndex])

        # Subscribers:
        self.odometry_subs = rospy.Subscriber('/hummingbird/ground_truth/odometry', Odometry, self.odometryCallback, queue_size=1)
        self.camera_subs = rospy.Subscriber('/uav/camera/left/image_rect_color', Image, self.rgbCameraCallback, queue_size=1)
        self.markers_subs = rospy.Subscriber('/uav/camera/left/ir_beacons', IRMarkerArray, self.irMarkersCallback, queue_size=1)
        self.uav_collision_subs = rospy.Subscriber('/uav/collision', std_Empty, self.droneCollisionCallback, queue_size=1 )

        # Publishers:
        self.trajectory_pub = rospy.Publisher('/hummingbird/command/trajectory', MultiDOFJointTrajectory,queue_size=1)
        self.dronePosePub = rospy.Publisher('/hummingbird/command/pose', PoseStamped, queue_size=1)
        self.rvizPath_pub = rospy.Publisher('/path', Path, queue_size=1)

        self.trajectorySamplingTimer = rospy.Timer(rospy.Duration(self.trajectorySamplingPeriod), self.timerCallback, oneshot=False, reset=False)
        time.sleep(1)
    ############################################# end of init function


    def computeGateVariables(self, gate):
        targetGateMarkersLocation = self.markersLocationDict[gate]
        targetGateDiagonalLength = np.max([np.abs(targetGateMarkersLocation[0, :] - marker) for marker in targetGateMarkersLocation[1:, :]])
        # used for drone traversing check
        self.targetGateHalfSideLength = targetGateDiagonalLength/(2 * math.sqrt(2)) * 1.1 # [m]
        self.targetGateNormalVector, self.targetGateCOM = computeGateNormalVector(targetGateMarkersLocation)

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
            # quaternion = ( pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
            # yaw = tf.transformations.euler_from_quaternion(quaternion)[2]
            # poseArray = np.array([pose.position.x, pose.position.y, pose.position.z, yaw])
            # self.benchamrkPoseDataBuffer.append(poseArray)

    def droneCollisionCallback(self, msg):
        print('drone collided!')

    def publishSampledPathRViz(self):
        now = rospy.Time.now()
        now_secs = now.to_sec()
        poses_list = []
        for i in range(self.max_ti_index):
            poseStamped_msg = PoseStamped()    
            poseStamped_msg.header.stamp = rospy.Time.from_sec(now_secs + i*self.trajectorySamplingPeriod)
            poseStamped_msg.header.frame_id = 'world'
            poseStamped_msg.pose.position.x = self.traj_position[i, 0]
            poseStamped_msg.pose.position.y = self.traj_position[i, 1]
            poseStamped_msg.pose.position.z = self.traj_position[i, 2]
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
        if self.ti_index == -1 or self.ti_index >= self.max_ti_index:
            return
        
        Pyaw = self.traj_orientation[self.ti_index] 
        q = Rotation.from_euler('z', Pyaw).as_quat()
        transform = Transform()
        transform.translation.x = self.traj_position[self.ti_index, 0]
        transform.translation.y = self.traj_position[self.ti_index, 1]
        transform.translation.z = self.traj_position[self.ti_index, 2]
        transform.rotation.x = q[0] 
        transform.rotation.y = q[1] 
        transform.rotation.z = q[2] 
        transform.rotation.w = q[3] 

        vel_twist = Twist()
        vel_twist.linear.x = self.traj_linearVel[self.ti_index, 0]
        vel_twist.linear.y = self.traj_linearVel[self.ti_index, 1]
        vel_twist.linear.z = self.traj_linearVel[self.ti_index, 2]
        vel_twist.angular.x = 0
        vel_twist.angular.y = 0
        vel_twist.angular.z = self.traj_angularVel[self.ti_index]

        Acc_twist = Twist()
        Acc_twist.linear.x = self.traj_linearAcc[self.ti_index, 0]
        Acc_twist.linear.y = self.traj_linearAcc[self.ti_index, 1]
        Acc_twist.linear.z = self.traj_linearAcc[self.ti_index, 2]
        Acc_twist.angular.x = 0
        Acc_twist.angular.y = 0
        Acc_twist.angular.z = self.traj_angularAcc[self.ti_index]

        self.ti_index += 1

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

    def compute_drone_targetGate_distance(self):
        position = self.lastOdomMsg.pose.pose.position
        dronePosition = np.array([position.x, position.y, position.z])
        Vdg = dronePosition - self.targetGateCOM
        return la.norm(Vdg)


    def irMarkersCallback(self, irMarkers_message):
        gatesMarkersDict = processMarkersMultiGate(irMarkers_message)
        targetGate = self.targetGateList[self.targetGateIndex]
        if targetGate in gatesMarkersDict.keys():
            markersData = gatesMarkersDict[targetGate]

            # check if all markers are visiable
            visiableMarkers = np.sum(markersData[:, -1] != 0)
            if  visiableMarkers <= 3:
                dis = self.compute_drone_targetGate_distance()
                print('targetGate: {}, dis: {}'.format(targetGate, dis))
                if dis <= 1.5: # observation
                    if self.targetGateIndex < len(self.targetGateList) - 1:
                        self.benchmarkTimerCount = 0
                        self.targetGateIndex += 1
                        self.computeGateVariables(self.targetGateList[self.targetGateIndex])
                    
                    print('target gate:', self.targetGateList[self.targetGateIndex])
                else:
                    print('not all markers are detected')
                return
            else:
                # all markers are visiable
                # print('found {} markers'.format(visiableMarkers))
                self.gateMarkers = markersData

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

        if (i != 0) and (curr_markers_tids[i] == tid) and (i >= self.numOfImageSequence-1):
            tid_sequence = curr_markers_tids[i-self.numOfImageSequence+1:i+1]

            # the tid diff is greater than 40ms, return None
            for k in range(self.numOfImageSequence-1):
                if tid_sequence[k+1] - tid_sequence[k] > 40:  
                    return None
            return tid_sequence

        return None

    def rgbCameraCallback(self, image_message):
        # cv_image = self.bridge.imgmsg_to_cv2(image_message, desired_encoding='bgr8')
        # if cv_image.shape != self.imageShape:
        #     rospy.logwarn('the received image size is different from what expected')
        #     #cv_image = cv2.resize(cv_image, (self.imageShape[1], self.imageShape[0]))
        # ts_rostime = image_message.header.stamp.to_sec()
        pass
    
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


    def run(self, startPose, endPose):
        # Place the drone:
        droneX, droneY, droneZ, droneYaw = startPose
        droneYaw = droneYaw * 180./np.pi

        self.placeDrone(droneX, droneY, droneZ, droneYaw)
        self.pauseGazebo()
        time.sleep(0.8)
        self.pauseGazebo(False)
        time.sleep(0.8)

        ## get the location of the features of the gate
        assert self.gateMarkers is not None

        featuresWorldPosition = self.markersLocationDict[self.targetGateList[0]][:2]

        print('featuresWorldPosition', featuresWorldPosition)

        ## Perform trajectory planning:
        ts = time.time()
        x_star, Px, Py, Pz, Pyaw = solve_n_D_OptimizationProblem(startPose, endPose, featuresWorldPosition, camera_FOV_h=80, camera_FOV_v=60)
        te = time.time()
        print('opt time =', te-ts)

        _, T_star = Px.applyOptVect(x_star)
        num = int(round(T_star/self.trajectorySamplingPeriod))
        t_list = np.linspace(0, T_star, num, endpoint=True)

        self.traj_position = np.zeros((num, 3))
        self.traj_position[:, 0] = Px.diP_dt(x_star, 0, t_list)
        self.traj_position[:, 1] = Py.diP_dt(x_star, 0, t_list)
        self.traj_position[:, 2] = Pz.diP_dt(x_star, 0, t_list) 

        self.traj_linearVel = np.zeros((num, 3))
        self.traj_linearVel[:, 0] = Px.diP_dt(x_star, 1, t_list)
        self.traj_linearVel[:, 1] = Py.diP_dt(x_star, 1, t_list)
        self.traj_linearVel[:, 2] = Pz.diP_dt(x_star, 1, t_list)

        self.traj_linearAcc = np.zeros((num, 3))
        self.traj_linearAcc[:, 0] = Px.diP_dt(x_star, 2, t_list)
        self.traj_linearAcc[:, 1] = Py.diP_dt(x_star, 2, t_list)
        self.traj_linearAcc[:, 2] = Pz.diP_dt(x_star, 2, t_list)

        self.traj_orientation = Pyaw.diP_dt(x_star, 0, t_list)
        self.traj_angularVel = Pyaw.diP_dt(x_star, 1, t_list)
        self.traj_angularAcc = Pyaw.diP_dt(x_star, 2, t_list)

        self.max_ti_index = num
        self.curr_sample_time = rospy.Time.now().to_sec()
        self.ti_index = 0

            
        self.publishSampledPathRViz()


        while not rospy.is_shutdown():
            pass

            # # markersData preprocessing 
            # tid_sequence = self.getMarkersDataSequence(self.t_id)
            # if tid_sequence is None:
            #     # rospy.logwarn('tid_sequence returned None')
            #     self.t_id = 0
            #     continue
            # markersDataSeq = []
            # for tid in tid_sequence:
            #     markersDataSeq.append(self.tid_markers_dict[tid]) 
            # currMarkersData = np.array(markersDataSeq)

            # self.t_id = 0

            # # twist data preprocessing
            # if len(self.twist_buff) < self.numOfTwistSequence:
            #     continue
            # currTwistData = np.array(self.twist_buff[-self.numOfTwistSequence:])

            # y_hat = self.networkInferencer.inference(currMarkersData, currTwistData)

            # positionCP, yawCP = y_hat[0][0].numpy(), y_hat[1][0].numpy()
            # positionCP = positionCP.reshape(5, 3).T
            # yawCP = yawCP.reshape(1, 3)
            # self.processControlPoints(positionCP, yawCP, currTime)


        # end of the for loop



def signal_handler(sig, frame):
    sys.exit(0)   

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    trajNav = TrajectoryNavigator()
    startPose = [-5, -30, 1, np.pi/3.]
    endPose = [0, -4, 2.5, np.pi/2.]
    trajNav.run(startPose, endPose)
