import ctypes

from tensorflow.python.keras.engine.training_utils import ModelInputs
libgcc_s = ctypes.CDLL('libgcc_s.so.1')
import sys 
import time
import numpy as np
import math
from scipy.spatial.transform import Rotation
import rospy
# import tf
from nav_msgs.msg import Path, Odometry
from trajectory_msgs.msg import MultiDOFJointTrajectory
from geometry_msgs.msg import PoseStamped
from gazebo_msgs.msg import ModelState
from sensor_msgs.msg import Image
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelState
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
# import tensorflow as tf
from tensorflow.keras import Input, layers, Model, backend as k
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.backend import dtype, expand_dims
from Bezier_untils import bezier4thOrder
from sympy import Symbol

class BezierPlannerSampler():
    def _getInceptionModel(self, inputShape):
        # local_weights_file = './inception_weights/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
        pre_trained_model = InceptionV3(input_shape = inputShape, 
                                        include_top = False, 
                                        weights = None)
        # pre_trained_model.load_weights(local_weights_file)
        return pre_trained_model
    
    def getDenseNet121Model(self, inputShape):
        pass
        # pre_trained_model = DenseNet121( input_shape = inputShape, weights="imagenet") 
        # return pre_trained_model

    def _createModel(self, inputShape=(240, 320, 3), pretrainedModelName='InceptionV3'):
        if pretrainedModelName == 'InceptionV3':
            pre_trained_model = self._getInceptionModel(inputShape)
            last_layer = pre_trained_model.get_layer('mixed7')
        elif pretrainedModelName == 'DenseNet121':
            pre_trained_model = self._getDenseNet121Model(inputShape)
            last_layer = pre_trained_model.get

        # process pre_trained_model
        for layer in pre_trained_model.layers:
            layer.trainable = False
        last_output = last_layer.output
        last_output_Flattened = layers.Flatten()(last_output)

        # process Twist data Input
        TwistInputLayer = Input(shape=(10, 4))
        twistFlatten = layers.Flatten()(TwistInputLayer)
        
        x = layers.concatenate([last_output_Flattened, twistFlatten])
        # Add a fully connected layer with 1,024 hidden units and ReLU activation
        x = layers.Dense(1024, activation='relu')(x)
        # Add a dropout rate of 0.2
        x = layers.Dropout(0.2)(x)  
        x = layers.Dense(512, activation='relu')(x)                
        # output layer:
        x = layers.Dense(14, activation=None)(x) 
        model = Model( [pre_trained_model.input, TwistInputLayer], x) 
        return model

    def __init__(self):
        rospy.init_node('bezier_planner_sampler_node', anonymous=True)
        # Twist variables
        self.twist_data_len = 10
        self.twist_buff_maxSize = self.twist_data_len*40
        self.twist_tid_list = []
        self.twist_buff = [] 
        # pose variables:
        self.poses_buff = []
        self.poses_buff_maxSize = self.twist_buff_maxSize
        
        # Bezier variables:
        self.t_sym = Symbol('t')

        # Deep learning stuff
        self.model = self._createModel() 
        self.model.load_weights('./learning/model_weights/weights20210708-151328.h5')

        # subs and pubs
        self.odometrySubs = rospy.Subscriber('/hummingbird/ground_truth/odometry', Odometry, self.__odometryCallback, queue_size=70)
        self.cameraSubs = rospy.Subscriber('/uav/camera/left_rgb_blurred/image_rect_color', Image, self.__rgbCameraCallback, queue_size=1)
        self.trajectoryPub = rospy.Publisher('hummingbird/command/trajectory', MultiDOFJointTrajectory, queue_size=1)
        self.dronePosePub = rospy.Publisher('/hummingbird/command/pose', PoseStamped, queue_size=1)
        self.rvizPath_pub = rospy.Publisher('/predicted_path', Path, queue_size=1)

        print('node initiated...')
        time.sleep(1)

    def __odometryCallback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        twist = msg.twist.twist
        t_id = int(msg.header.stamp.to_sec()*1000)
        twist_data = np.array([twist.linear.x, twist.linear.y, twist.linear.z, twist.angular.z])
        self.twist_tid_list.append(t_id)
        self.twist_buff.append(twist_data)
        self.poses_buff.append(msg.pose.pose)
        if len(self.twist_buff) > self.twist_buff_maxSize:
            self.twist_buff = self.twist_buff[-self.twist_buff_maxSize :]
            self.twist_tid_list = self.twist_tid_list[-self.twist_buff_maxSize :]
            self.poses_buff = self.poses_buff[-self.poses_buff_maxSize :]


    def __getCurrentPoseAndTwistDataList(self, t_id):
        curr_tid_nparray  = np.array(self.twist_tid_list)
        curr_twist_nparry = np.array(self.twist_buff)
        curr_pose_nparray = np.array(self.poses_buff)
        idx = np.searchsorted(curr_tid_nparray, t_id, side='left')
        # check if idx is not out of range or is not the last element in the array (there is no upper bound)
        # take the data from the idx [inclusive] back to idx-self.twist_data_len [exclusive]
        if idx <= self.twist_buff_maxSize-2 and idx-self.twist_data_len+1>= 0:
            currPose = curr_pose_nparray[idx]
            return currPose, curr_twist_nparry[idx-self.twist_data_len+1:idx+1]
        else:
            print('twist data returned None')
            return None, None

    def __transfromPositionControlPoints(self, points, pose):
        # compute the rotation matrix that rotates points on the Z axis only (yaw):
        q = pose.orientation
        r = Rotation.from_quat([q.x, q.y, q.z, q.w])
        rpy = r.as_euler('xyz', degrees=False)
        rotationMatrix = Rotation.from_euler('z', rpy[2], degrees=False).as_matrix()
        # compute the translation matrix:
        translationMatrix = np.array([pose.position.x, pose.position.y, pose.position.z]).reshape((3, 1))
        # compute the transformation matrix:
        transformationMatrix = np.concatenate([rotationMatrix, translationMatrix], axis=1)
        lastRow = np.array([0, 0, 0, 1]).reshape((1, 4))
        transformationMatrix = np.concatenate([transformationMatrix, lastRow], axis=0)
        # add a row of 1 to all the points:
        points = np.concatenate([points, np.ones((1, points.shape[1]))], axis=0)
        # transfrom the points:
        transformedPoints = np.matmul(transformationMatrix, points)
        # return the points after removing the last row (all ones):
        return transformedPoints[:-1, :]

    def __rgbCameraCallback(self, msg):
        tid = int(msg.header.stamp.to_sec()*1000)
        currPose, twistData = self.__getCurrentPoseAndTwistDataList(tid)
        if twistData is None:
            return
        
        image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        image = cv2.resize(image, (320, 240))
        image = np.expand_dims(image, axis=0)

        twistData = np.expand_dims(twistData, axis=0)
        self.modelInput = [image, twistData]
        # controlPoints = self.model.predict(modelInput)
        controlPoints = self.model(self.modelInput, training=False)[0]
        ts = time.time()
        positionControlPoints = controlPoints[:12]
        positionControlPoints = np.reshape(positionControlPoints, (4, 3))
        # add the first position control point (all zeros)
        positionControlPoints = np.concatenate( [np.zeros((1, 3)), positionControlPoints], axis=0)
        positionControlPoints = positionControlPoints.T
        positionControlPoints = self.__transfromPositionControlPoints(positionControlPoints, currPose)
        # processing the yaw control points:
        # yawControlPoints = controlPoints[-2:].reshape()
        # yawControlPoints = 
        self.__publishPath(positionControlPoints)
        te = time.time()
        print('duration:', (te-ts), 1/(te-ts))

    def __publishPath(self, cp):
        assert cp.shape == (3, 5), 'assertion failed, the shape of the position control points is not correct'
        acc = 20
        endTime = 1
        t_space = np.linspace(0, endTime, acc)
        poses_list = []
        tstart = rospy.Time.now()
        for ti in t_space:
            poseStamped_msg = PoseStamped()    
            poseStamped_msg.header.stamp = rospy.Time.from_sec(tstart.to_sec() + ti)
            poseStamped_msg.header.frame_id = 'world'
            Pxyz = bezier4thOrder(cp, ti)
            poseStamped_msg.pose.position.x = Pxyz[0]
            poseStamped_msg.pose.position.y = Pxyz[1]
            poseStamped_msg.pose.position.z = Pxyz[2]
            quat = [0, 0, 0, 1]
            poseStamped_msg.pose.orientation.x = quat[0]
            poseStamped_msg.pose.orientation.y = quat[1]
            poseStamped_msg.pose.orientation.z = quat[2]
            poseStamped_msg.pose.orientation.w = quat[3]
            poses_list.append(poseStamped_msg)
        path = Path()
        path.poses = poses_list        
        path.header.stamp = tstart
        path.header.frame_id = 'world'
        self.rvizPath_pub.publish(path)


    def placeDrone(self, x, y, z, yaw=-1, qx=0, qy=0, qz=0, qw=1):
        # if yaw is provided (in degrees), then caculate the quaternion
        if yaw != -1:
            # q = tf.transformations.quaternion_from_euler(0, 0, yaw*math.pi/180.0) 
            q = (Rotation.from_euler('xyz', [0, 0, yaw], degrees=True)).as_quat()
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
        droneX, droneY, droneZ, droneYaw = self.__generateRandomPose(gateX, gateY, gateZ)
        self.placeDrone(droneX, droneY, droneZ, droneYaw)
        self.pauseGazebo()
        time.sleep(0.8)
        self.pauseGazebo(False)

def main():
    bps = BezierPlannerSampler()
    bps.run()
    rospy.spin()

if __name__ == '__main__':
    main()