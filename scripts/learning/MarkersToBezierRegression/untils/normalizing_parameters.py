
import sys

from matplotlib.pyplot import axis
# sys.path.append('../../')
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import os
import numpy as np
import cv2
import pandas as pd


def compute_twistData_maxValues(directory):
    '''
        finds the max parameters for twistData 
        twistData shape is (B, N, 4), B is the batch, N is the length of the vector,
        the last 4 is for vel_x, vel_y, vel_z, vel_yaw.
    '''
    df = pd.read_pickle(directory)
    twistData = np.array(df['vel'].tolist())
    exp_shape = (100, 4)

    assert twistData.shape[1:] == exp_shape, "twistData shape is not as expected, exp: {}, found: {}".format(exp_shape, twistData.shape)
    print(twistData.shape)

    twistData1 = twistData.reshape(-1, 4)
    max_twistData = (twistData1).max(axis=0)
    min_twistData = (twistData1).min(axis=0)
    for i, axis in enumerate(['x', 'y', 'z', 'yaw']):
        print('vel_{}, max={}, min={}'.format(axis, max_twistData[i], min_twistData[i]))

    print("number of negative vel_x =", twistData1[twistData1[:, 0] < 0].min(axis=0))
    
    # twistData_normalized = np.multiply(twistData, 1./max_twistData)
    # print(twistData_normalized.shape)
    # print(twistData_normalized.reshape(-1, 4).max(axis=0), twistData_normalized.reshape(-1, 4).mean(axis=0))

    
    return max_twistData



def main():
    # allDataFileWithMarkers = '/home/majd/catkin_ws/src/basic_rl_agent/data/imageBezierData1/allData_imageBezierData1_20220304-1535.pkl'
    allDataFileWithMarkers = '/home/majd/catkin_ws/src/basic_rl_agent/data2/flightgoggles/DataWithRatio_imageBezierData1_20220304-1756.pkl'
    max_twist_val = compute_twistData_maxValues(allDataFileWithMarkers)
    norm_factor = 1. / max_twist_val
    print(np.multiply(max_twist_val, norm_factor))


if __name__ == '__main__':
    main()