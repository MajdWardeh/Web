# import sys
# ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
# if ros_path in sys.path:
#     sys.path.remove(ros_path)
from Bezier_untils import BezierVisulizer
import sys
import os
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
from scipy.spatial.transform import Rotation
import math
from math import floor
import rospy
import roslaunch
# import tf
from std_msgs.msg import Empty as std_Empty
from geometry_msgs.msg import PoseStamped, Pose, Quaternion, Transform, Twist
from gazebo_msgs.msg import ModelState, LinkStates
from flightgoggles.msg import IRMarkerArray
from nav_msgs.msg import Path, Odometry
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
from sensor_msgs.msg import Image
import dynamic_reconfigure.client
# import cv2
from IrMarkersUtils import processMarkersMultiGate 

from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelState

from learning.MarkersToBezierRegression.FullyConnectedMarkersDataToBezierRegressor_withYawAndTwistData_configurable import Network
from Bezier_untils import bezier4thOrder, bezier2ndOrder, bezier3edOrder, bezier1stOrder

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
        self.markersDataFactor = np.array([1.0/float(self.imageShape[1]), 1.0/float(self.imageShape[0]), 1.0])
        
        self.t_id = 0

        self.model = Network(networkConfig).getModel()
        self.model.load_weights(weightsFile)

        self.bezierVisualizer = BezierVisulizer(plot_delay=0.1)
        self.lastIrMarkersMsgTime = None
        self.IrMarkersMsgIntervalSum = 0
        self.IrMarerksMsgCount_FPS = 0
        self.irMarkersMsgCount = 0

        # benchmark variables:
        self.benchmarkCheckFreq = 10
        self.TIMEOUT_SEC = 15 
        self.roundTimeOutCount = self.TIMEOUT_SEC * self.benchmarkCheckFreq # sec
        self.benchmarking = False
        self.roundTimedOut = False
        self.benchmarkTwistDataBuffer = []
        self.benchmarkTimer = None # rospy Timer pointer
        self.benchmarkTimerCount = 0
        self.droneCollided = False
        self.dronePassedGated6 = False
        self.benchmarkResultsDict = {
            'pose': [],
            'round_success': [],
            'round_finish_reason': [],
            'average_twist': [],
            'peak_twist': [],
            'average_FPS': [],
            'target_FPS': []
        }

        # the location of the gate in FG V2.04 
        self.gate6CenterWorld = np.array([-10.04867002, 30.62322557, 2.8979407]).reshape(3, 1)
        self.gate6PassedPointWorld = self.gate6CenterWorld.reshape(3, ) + np.array([0, 0.30, 0])
        self.gate6PassingThreshold = 0.5

        # ir_beacons variables
        self.targetGate = 'Gate6'
       
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
        self.benchmarTimer = rospy.Timer(rospy.Duration(1/self.benchmarkCheckFreq), self.benchmarkTimerCallback, oneshot=False, reset=False)
        time.sleep(1)
    
    def odometryCallback(self, msg):
        self.lastOdomMsg = msg
        t_id = int(msg.header.stamp.to_sec()*1000)
        twist = msg.twist.twist
        twist_data = np.array([twist.linear.x, twist.linear.y, twist.linear.z, twist.angular.z])
        self.twist_tid_list.append(t_id)
        self.twist_buff.append(twist_data)
        if len(self.twist_buff) > self.twist_buff_maxSize:
            self.twist_buff = self.twist_buff[-self.twist_buff_maxSize :]
            self.twist_tid_list = self.twist_tid_list[-self.twist_buff_maxSize :]     
        if self.benchmarking:
            self.benchmarkTwistDataBuffer.append(twist_data)

    def droneCollisionCallback(self, msg):
        if self.benchmarking:
            print('drone collided! round finished')
            self.droneCollided = True
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

        t = (rospy.Time.now().to_sec() - currTime)/self.T
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

        self.trajectory_pub.publish(trajectory)
        
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
        self.publishSampledPathRViz(positionCP_world)

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
                print('not all markers are detected')
                return
            else:
                print('found {} markers'.format(visiableMarkers))
                self.markersData = markersData
                self.currTime = rospy.Time.now()
                self.t_id = int(irMarkers_message.header.stamp.to_sec()*1000)

        else:
                print('no markers were found')
                self.lastIrMarkersMsgTime = None

        if self.lastIrMarkersMsgTime is None:
            self.lastIrMarkersMsgTime = irMarkers_message.header.stamp.to_sec()
            return
        msgTime = irMarkers_message.header.stamp.to_sec() 
        self.IrMarkersMsgIntervalSum += msgTime - self.lastIrMarkersMsgTime
        self.IrMarerksMsgCount_FPS += 1
        self.lastIrMarkersMsgTime = msgTime
        print('average FPS = ', self.IrMarerksMsgCount_FPS/self.IrMarkersMsgIntervalSum)

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

    def generateRandomPose(self, gateX, gateY, gateZ, maxYawRotation=35):
        xmin, xmax = gateX - 3, gateX + 3
        ymin, ymax = gateY - 7, gateY - 12
        zmin, zmax = gateZ - 0.8, gateZ + 0.5
        x = xmin + np.random.rand() * (xmax - xmin)
        y = ymin + np.random.rand() * (ymax - ymin)
        z = zmin + np.random.rand() * (zmax - zmin)
        # yaw = np.random.normal(90, maxYawRotation/5) # 99.9% of the samples are in 5*segma
        if np.random.rand() > 0.5:
            yawMin, yawMax = 60, 70
        else:
            yawMin, yawMax = 110, 120
        yaw = yawMin + np.random.rand() * (yawMax-yawMin)
        return x, y, z, yaw

    def benchmarkTimerCallback(self, timerMsg):
        if not self.benchmarking:
            return 

        # check if the drone arrived to the gate:
        position = self.lastOdomMsg.pose.pose.position
        dronePosition = np.array([position.x, position.y, position.z])
        diff = dronePosition - self.gate6PassedPointWorld
        normOfDiff = la.norm(diff)
        if normOfDiff < self.gate6PassingThreshold:
            self.dronePassedGated6 = True
            self.benchmarkTimerCount = 0
            self.roundFinished = True
            self.benchmarking = False
            print('drone passed the gate!')

        self.benchmarkTimerCount += 1
        if self.benchmarkTimerCount >= self.roundTimeOutCount:
            print('timeout! round finished.')
            self.roundTimedOut = True
            self.roundFinished = True
            self.benchmarking = False
        

    def run(self, poses):
        '''
            @param poese: a list of np arraies. each np array has an initial pose (x, y, z, yaw).

            each pose with a target_FPS correspond to a round.
            The round is finished if the drone reached the gate or if the roundTimeOut accured or if the drone is collided.
        '''
        print('Benchmark Started')
        for roundId, pose in enumerate(poses[:7]):
            print('\nprocessing round {} ##########'.format(roundId))
            # Place the drone:
            droneX, droneY, droneZ, droneYaw = pose
            self.curr_trajectory = None

            self.placeDrone(droneX, droneY, droneZ, droneYaw)
            self.pauseGazebo()
            time.sleep(0.8)
            self.pauseGazebo(False)
            time.sleep(0.8)

            # variables preparation for a new round:
            self.droneCollided = False
            self.dronePassedGated6 = False
            self.roundTimedOut = False
            self.IrMarkersMsgIntervalSum = 0
            self.IrMarerksMsgCount_FPS = 0
            self.benchmarkTwistDataBuffer = [] 

            self.roundFinished = False
            self.benchmarkTimerCount = 0
            self.benchmarking = True

            while not rospy.is_shutdown() and not self.roundFinished:

                # check if there are new markers data
                if self.t_id != 0:
                    # save current time
                    currTime = self.currTime

                    self.t_id = 0

                    twistData = np.concatenate([self.twist_buff[-1], self.twist_buff[-2]], axis=0)
                    twistData = twistData[np.newaxis, :]

                    # prepare markersData
                    markersDataNormalized = np.multiply(self.markersData, self.markersDataFactor)
                    markersDataNormalized = markersDataNormalized.reshape(12, )
                    markersDataNormalized = markersDataNormalized[np.newaxis, :]


                    networkInput = [markersDataNormalized, twistData]
                    # network inferencing:
                    y_hat = self.model(networkInput, training=False)
                    positionCP, yawCP = y_hat[0][0].numpy(), y_hat[1][0].numpy()
                    positionCP = positionCP.reshape(5, 3).T
                    yawCP = yawCP.reshape(1, 3)
                    self.imageToPlot = np.zeros(shape=self.imageShape)
                    self.positionCP = positionCP
                    self.yawCP = yawCP
                    self.processControlPoints(positionCP, yawCP, currTime)

                # check if the drone reached
    
            # process and save benchmark data:
            if self.roundFinished:
                self.benchmarking = False

                if self.dronePassedGated6:
                    roundFinishReason = 'dronePassedGate'
                elif self.droneCollided:
                    roundFinishReason = 'droneCollided'
                elif self.roundTimedOut:
                    roundFinishReason = 'roundTimedOut'
                else:
                    raise Exception('roundFinishedReason is unknown to me!')

                # peak and average speed:
                self.benchmarkTwistDataBuffer = np.array(self.benchmarkTwistDataBuffer)
                averageTwist = np.mean(self.benchmarkTwistDataBuffer, axis=0)
                linearVel = self.benchmarkTwistDataBuffer[:, :-1] # remove the angular yaw velocity
                linearVel = la.norm(linearVel, axis=1) 
                print(linearVel.shape)
                peakTwist = np.max(linearVel)

                self.benchmarkResultsDict['pose'].append(pose)
                self.benchmarkResultsDict['round_success'].append(self.dronePassedGated6)
                self.benchmarkResultsDict['round_finish_reason'].append(roundFinishReason)
                self.benchmarkResultsDict['average_twist'].append(averageTwist)
                self.benchmarkResultsDict['peak_twist'].append(peakTwist)
                # self.benchmarkResultsDict['average_FPS'].append(self.IrMarerksMsgCount_FPS/self.IrMarkersMsgIntervalSum)
                # self.benchmarkResultsDict['target_FPS'].append()


    def benchmark(self, benchmarkPosesRootDir):
        posesFiles = os.listdir(benchmarkPosesRootDir)
        for file in posesFiles:
            posesDataFrame = pd.read_pickle(os.path.join(benchmarkPosesRootDir, file))
            poses = posesDataFrame['poses'].tolist()
            self.run(poses)

    def generateBenchmarkPosesFile(self, fileName, numOfPoses):
        gateX, gateY, gateZ = self.gate6CenterWorld.reshape(3, )
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

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    config6 = {
        'numOfDenseLayers': 3,
        'numOfUnitsPerLayer': [100, 80, 50],
        'dropRatePerLayer': [0, 0, 0], 
        'learningRate': 0.0005,
        'configNum': 6,
        'numOfEpochs': 1200
    }
    weightsFile = '/home/majd/catkin_ws/src/basic_rl_agent/data/deep_learning/MarkersToBezierDataFolder/models_weights/weights_MarkersToBeizer_FC_scratch_withYawAndTwistData_config6_20210810-044932.h5'
    networkBenchmarker = NetworkNavigatorBenchmarker(networkConfig=config6, weightsFile=weightsFile)
    
    benchmarkPosesRootDir = '/home/majd/catkin_ws/src/basic_rl_agent/data/deep_learning/benchmarks/benchmarkPosesFiles'

    # networkBenchmarker.generateBenchmarkPosesFile(fileName=os.path.join(benchmarkPosesRootDir, 'benchmarkerPosesFile1.pkl'), numOfPoses=10) 
    
    networkBenchmarker.benchmark(benchmarkPosesRootDir)



