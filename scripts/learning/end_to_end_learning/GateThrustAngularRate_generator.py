import os
import math
import numpy as np
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import pandas as pd

from tensorflow.keras.utils import Sequence


sys.path.append('/home/majd/catkin_ws/src/basic_rl_agent/scripts/learning')
sys.path.append('/home/majd/catkin_ws/src/basic_rl_agent/scripts/learning/gateCornerLocalization')


class GateCommandGenerator(Sequence):
    '''
        A custom data generator to generate training/validation/testing samples.
    '''
    def __init__(self, df, config={}, keep_in_ram=False):
        self.imageSequenceList = df['images'].tolist()
        self.commandSequenceList = df['commandSequence'].tolist()
        self.__process_config(config)

        self.keep_in_ram = keep_in_ram
        if self.keep_in_ram:
            self.lookup_return_list = [None] * self.__len__()

        
    def __process_config(self, config={}):
        self.batch_size = config.get('batch_size', 32)
        target_size = config.get('target_size', None)
        self.target_width = target_size[1]
        self.target_height = target_size[0]
        self.r_range = config.get('r_range', (0.2, 20.))
        cam_fov = 90*0.85  # in degrees -- needs to be a bit smaller than 90 in fact because of cone vs. square
        alpha = cam_fov / 180.0 * np.pi / 2.0  # alpha is half of fov angle in rads
        self.theta_range = config.get('theta_range', (-alpha, alpha))
        self.phi_range = config.get('phi_range', (-np.pi, np.pi))
        eps = np.pi * 0.001
        self.yaw_range = config.get('yaw_range', (-np.pi/2.+eps, np.pi/2.-eps))
    
    def __len__(self):
        return math.ceil(len(self.imageSequenceList) / self.batch_size)

    def normalize_gate(self, gate_center_sph, gate_heading):
        r_norm = 2.0 * (gate_center_sph[0] - self.r_range[0])/(self.r_range[1]-self.r_range[0]) - 1.0
        theta_norm = 2.0 * (gate_center_sph[1] - self.theta_range[0])/(self.theta_range[1]-self.theta_range[0]) - 1.0
        phi_norm = 2.0 * (gate_center_sph[2] - self.phi_range[0])/(self.phi_range[1]-self.phi_range[0]) - 1.0
        yaw_norm = 2.0 * (gate_heading - self.yaw_range[0])/(self.yaw_range[1]-self.yaw_range[0]) - 1.0
        return np.array([r_norm, theta_norm, phi_norm, yaw_norm])

    def __loaditem__(self, index):
        X_batch = [] # images (normalized between -1 and 1)
        y_batch = [] # thrust and angluar rates command
        for row in range(min(self.batch_size, len(self.imageSequenceList)-index*self.batch_size)):
            imageSequence = self.imageSequenceList[index*self.batch_size + row]
            # take the last image in the image sequence
            image_name = imageSequence[-1, 0]
            img = cv2.imread(image_name) # color channels order is BGR
            img = cv2.resize(img, (self.target_width, self.target_height))

            # take the first command in the command squence
            commandSequence = self.commandSequenceList[index*self.batch_size + row]
            command = commandSequence[:, 0]

            X_batch.append(img)
            y_batch.append(command)

        X_batch = np.array(X_batch).astype(np.float32)
        X_batch = 2.0 * (X_batch / 255.0) - 1.0
        y_batch = np.array(y_batch)

        return (X_batch, y_batch)

    def __getitem__(self, index):
        if self.keep_in_ram:
            return_value = self.lookup_return_list[index]
            if return_value is None:
                return_value = self.__loaditem__(index)
                self.lookup_return_list[index] = return_value
        else:
            return_value = self.__loaditem__(index)
        return return_value 



def testing():
    df_path = '/home/majd/catkin_ws/src/basic_rl_agent/data2/flightgoggles/datasets/imageLowLevelControl/allData_imageLowLevelControl_rowsCount365601_20220412-1704.pkl'
    df = pd.read_pickle(df_path)

    config = {
        'batch_size': 1, 
        'target_size': (240, 320)
    }
    gateCommandGenerator = GateCommandGenerator(df, config, keep_in_ram=False)


    for batch_idx in range(gateCommandGenerator.__len__())[:]:
        X_batch, y_batch = gateCommandGenerator.__getitem__(batch_idx)
        for i in range(len(y_batch)):
            img = X_batch[i]
            assert img is not None
            img = ( (img + 1.)/2.) * 255.
            img = img.astype(np.uint8)
            cv2.imshow('image', img)
            thrustAngularRatesCommand = y_batch[i]
            print(thrustAngularRatesCommand)
            if cv2.waitKey(0) == ord('q'):
                break
    

if __name__ == '__main__':
    testing()


