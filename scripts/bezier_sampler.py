import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')
import math
import time
import numpy as np
import rospy
import sys 
from nav_msgs.msg import Odometry
from trajectory_msgs.msg import MultiDOFJointTrajectory
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelState
from cv_bridge import CvBridge
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import tensorflow as tf
# from tensorflow.keras import Input, layers, Model, backend as k
# from tensorflow.keras.applications.inception_v3 import InceptionV3
# from tensorflow.python.keras.backend import expand_dims

class BezierPlannerSampler():
    def _getInceptionModel(self):
        local_weights_file = './inception_weights/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
        pre_trained_model = InceptionV3(input_shape = (240, 320, 3), 
                                        include_top = False, 
                                        weights = None)
        pre_trained_model.load_weights(local_weights_file)

        for layer in pre_trained_model.layers:
            layer.trainable = False

        last_layer = pre_trained_model.get_layer('mixed7')
        # print('last layer output shape: ', last_layer.output_shape)
        last_output = last_layer.output

        TwistInputLayer = Input(shape=(10, 4))

        x = layers.Flatten()(last_output)

        twistFlatten = layers.Flatten()(TwistInputLayer)
        
        x = layers.concatenate([x, twistFlatten])
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
        # variables
        # self.bridge = CvBridge()
        self.twist_data_len = 10
        self.twist_buff_maxSize = self.twist_data_len*5
        self.twist_tid_list = []
        self.twist_buff = [] 

        # Deep learning stuff
        # self.model = self._getInceptionModel() 
        # self.model.load_weights('./learning/model_weights/weights20210707-183902.h5')

        # subs and pubs
        self.odometrySubs = rospy.Subscriber('/hummingbird/ground_truth/odometry', Odometry, self.__odometryCallback, queue_size=70)
        self.cameraSubs = rospy.Subscriber('/uav/camera/left_rgb_blurred/image_rect_color', Image, self.__rgbCameraCallback, queue_size=2)
        self.trajectoryPub = rospy.Publisher('hummingbird/command/trajectory', MultiDOFJointTrajectory, queue_size=1)

        print('node initiated...')

    def __odometryCallback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        twist = msg.twist.twist
        t_id = int(msg.header.stamp.to_sec()*1000)
        twist_data = np.array([twist.linear.x, twist.linear.y, twist.linear.z, twist.angular.z])
        self.twist_tid_list.append(t_id)
        self.twist_buff.append(twist_data)
        if len(self.twist_buff) > self.twist_buff_maxSize:
            self.twist_buff = self.twist_buff[-self.twist_buff_maxSize :]
            self.twist_tid_list = self.twist_tid_list[-self.twist_buff_maxSize :]


    def __computeTwistDataList(self, t_id):
        curr_tid_nparray  = np.array(self.twist_tid_list)
        curr_twist_nparry = np.array(self.twist_buff)
        idx = np.searchsorted(curr_tid_nparray, t_id, side='left')
        # check if idx is not out of range or is not the last element in the array (there is no upper bound)
        # take the data from the idx [inclusive] back to idx-self.twist_data_len [exclusive]
        if idx <= self.twist_buff_maxSize-2 and idx-self.twist_data_len+1>= 0:
            return curr_twist_nparry[idx-self.twist_data_len+1:idx+1]
        return None

    def __rgbCameraCallback(self, msg):
        print('image', msg.header.stamp.to_sec())
        # do NN inferencing
        image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        # image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        image = cv2.resize(image, (240, 320))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tid = int(msg.header.stamp.to_sec()*1000)
        twistData = self.__computeTwistDataList(tid)
        modelInput = np.array([image, twistData])
        modelInput = np.expand_dims(modelInput, axis=-1)
        # controlPoints = self.model.predict(modelInput)
        # print(controlPoints[0].shape)

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
        return x, y, z, ya

    def run(self, gateX=30.7, gateY=10, gateZ=2.4):
        droneX, droneY, droneZ, droneYaw = self.__generateRandomPose(gateX, gateY, gateZ)
        self.placeDrone(droneX, droneY, droneZ, droneYaw)
        self.pauseGazebo()
        time.sleep(0.8)
        self.pauseGazebo(False)

def main():
    bps = BezierPlannerSampler()
    rospy.spin()

if __name__ == '__main__':
    main()