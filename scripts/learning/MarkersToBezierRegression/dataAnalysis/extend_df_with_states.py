import sys
import warnings

from numpy import random
from numpy.lib.function_base import average
sys.path.append('../../')
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import datetime
import numpy as np
import numpy.linalg as la
import math
import cv2
import pandas as pd
import tensorflow as tf
from scipy.spatial.transform.rotation import Rotation as Rot
from learning.gateCornerLocalization.markers_reprojection import camera_info, compute_markers_3d_position

def get_markers_3d_world():
    return np.array([[0.9014978, 0.001497865, 2.961498], [-0.898504, 0.001497627, 2.961498], [0.9014994, 0.001497865, 1.115498], [-0.898504, 0.001497627, 1.115498]])

def get_T_b_c():
    R_b_c = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]) # rotaiton from frame B to frame C
    T_b_c = np.zeros((4, 4))
    T_b_c[:3, :3] = R_b_c
    T_b_c[3, 3] = 1
    return T_b_c

def compute_T_w_c(markers, markers_3d_w, k_camera, dist_coeffs=np.zeros((4, 1))):
    '''
        compute the transformation from the world frame W to the camera frame C.
        @arg markers: np array of shape (4, 3), each row has u, v, Z. where u and v are the pixel locations of each marker. Z is its depth (in meters or so not sure!).
        @arg markers_3d_w: np array of shape (4, 3) each row has X, Y, Z, the 3d location of the gate's markers in the world frame.
        @arg k_camera: the camera intrinsics matrix.
    '''
    markers_2d = markers[:, 0:2].astype(np.float32)
    success, rotation_vector, translation_vector = cv2.solvePnP(markers_3d_w, markers_2d, k_camera, dist_coeffs, flags=cv2.SOLVEPNP_IPPE) # flags=cv2.SOLVEPNP_P3P
    # assert success, 'the solvePnP method was not successful.'
    if success:
        R_c_w, _ = cv2.Rodrigues(rotation_vector)
        T_c_w = np.zeros((4, 4))
        T_c_w[:3, :3] = R_c_w
        T_c_w[:3, 3] = translation_vector.reshape(-1,)
        T_c_w[3, 3] = 1
        return  la.inv(T_c_w)
    else:
        warnings.warn('the solvePnP method was not successful.')
        return None

def compute_T_w_b(T_w_c, T_b_c):
    T_w_b = np.matmul(T_w_c, la.inv(T_b_c))
    return T_w_b


def test_compute_T_w_c(directory):
    df = pd.read_pickle(directory)
    markersList = df['markersData'].tolist()

    markers = markersList[0][-1]
    markers_3d_w = get_markers_3d_world()
    k_camera = camera_info()
    T_w_c = compute_T_w_c(markers, markers_3d_w, k_camera)

    fx, fy, cx, cy = k_camera[0, 0], k_camera[1, 1], k_camera[0, -1], k_camera[1, -1]
    markers_3d_c = compute_markers_3d_position(markers, fx, fy, cx, cy)
    markers_3d_c_hmg = np.ones((4, markers_3d_c.shape[1]))
    markers_3d_c_hmg[:3, :] = markers_3d_c 

    markers_reprojected_3d_w = np.matmul(T_w_c, markers_3d_c_hmg)[:3, :]
    reporjection_error = np.square(markers_reprojected_3d_w - markers_3d_w.T).mean()
    print('reprojection error:', reporjection_error)


def add_drone_state_to_df(directory):
    df = pd.read_pickle(directory)
    # df = df.sample(frac=1)
    # imagesList = df['images'].tolist()
    markersList = df['markersData'].tolist()
    # positionCP_list = df['positionControlPoints'].tolist()
    # yawCP_list = df['yawControlPoints'].tolist()

    T_c_b = la.inv(get_T_b_c())
    k_camera = camera_info()
    markers_3d_w = get_markers_3d_world()

    states_sequence_list = []
    unsucessfull_T_w_c_counter = 0
    for markers_sequence in markersList:
        states = []
        for markers in markers_sequence:
            if np.sum(markers[:, -1] != 0, axis=0)==4:
                T_w_c = compute_T_w_c(markers, markers_3d_w, k_camera)
                ## compute reprojection error:
                fx, fy, cx, cy = k_camera[0, 0], k_camera[1, 1], k_camera[0, -1], k_camera[1, -1]
                markers_3d_c = compute_markers_3d_position(markers, fx, fy, cx, cy)
                markers_3d_c_hmg = np.ones((4, markers_3d_c.shape[1]))
                markers_3d_c_hmg[:3, :] = markers_3d_c 

                markers_reprojected_3d_w = np.matmul(T_w_c, markers_3d_c_hmg)[:3, :]
                reprojection_error = float(np.square(markers_reprojected_3d_w - markers_3d_w.T).mean())
                if reprojection_error > 1e-6:
                    warnings.warn('large reprojection error, {}'.format(reprojection_error))

                T_w_b = np.matmul(T_w_c, T_c_b)
                R_w_b = T_w_b[:3, :3]
                t_w_b = T_w_b[:3, 3].tolist()
                rpy = Rot.from_dcm(R_w_b).as_euler('XYZ', degrees=True).tolist()
                state = t_w_b + rpy
                states.append(np.array(state))
            else:
                unsucessfull_T_w_c_counter += 1
                states.append(np.array([None]*6))
        states_sequence_list.append(np.array(states))
    
    print('len of statesList: ', len(states_sequence_list))
    print('num of unsuccessful PnP trails', unsucessfull_T_w_c_counter)
    df['statesList'] = states_sequence_list
    return df
    

def main():
    working_dir = '/home/majd/catkin_ws/src/basic_rl_agent/data2/flightgoggles/datasets/imageBezierDataV2_1'
    allDataFileWithMarkers = 'allData_imageBezierDataV2_1_20220407-1358.pkl'

    df_with_states = add_drone_state_to_df(os.path.join(working_dir, allDataFileWithMarkers))

    index = allDataFileWithMarkers.find('_')
    df_new_name = allDataFileWithMarkers[:index] + '_WITH_STATES' + allDataFileWithMarkers[index:]
    
    df_with_states.to_pickle(os.path.join(working_dir, df_new_name))
    print('{} was saved.'.format(os.path.join(working_dir, df_new_name)))



if __name__ == '__main__':
    main()
    