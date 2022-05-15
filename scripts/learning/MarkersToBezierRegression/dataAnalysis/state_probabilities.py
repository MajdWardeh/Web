import sys
import warnings

from numpy import random
sys.path.append('../../')
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import datetime
import time
import numpy as np
import numpy.linalg as la
import math
import cv2
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial.transform.rotation import Rotation as Rot
from learning.gateCornerLocalization.markers_reprojection import camera_info, compute_markers_3d_position

class StateProbability:

    def __init__(self, path):
        self.get_hard_coded_data()
        self.df = pd.read_pickle(path)
        statesList = self.df['statesList'].tolist()
        full_states_filtered = []
        for states_sequence in statesList:
            states = states_sequence[-1]
            if (states == None).all():
                continue
            full_states_filtered.append(states)
        full_states_filtered = np.array(full_states_filtered, dtype=np.float32)
        print(full_states_filtered.shape)
        self.states = full_states_filtered[:, [0, 1, 2, 5]]

        ## check if the hard coded min max are still valid
        self.check_min_max_values()

        ## compute states bin edges:
        self.states_bin_edges = []
        for i in range(self.states.shape[1]):
            min_v = self.states_min_max_values[i, 0]
            max_v = self.states_min_max_values[i, 1]
            num = int(round( (max_v - min_v) / self.states_resolution[i]))
            bin_edges_i = np.linspace(min_v, max_v, num+1, endpoint=True)
            self.states_bin_edges.append(bin_edges_i)

        self.x_hist, x_bin_edges = np.histogram(self.states[:, 0], self.states_bin_edges[0], density=True)

        bins = [bin_ for bin_ in self.states_bin_edges]
        self.hist_dd, _ = np.histogramdd(self.states, bins)
        # print(x_bin_edges)
        # print(self.x_hist)

        # plt.figure()
        # plt.hist(x_bin_edges[:-1], x_bin_edges, weights=self.x_hist)
        # plt.show()
    
    def get_hard_coded_data(self):
        self.states_resolution = np.array([0.1, 0.1, 0.1, 1.])
        self.states_min_max_values = np.array([
           (-8, 8),
           (-24, -1),
           (0.8, 4),
           (45, 135)
        ], dtype=np.float32)
    
    def check_min_max_values(self):
        states_computed_min = self.states.min(axis=0)
        states_computed_max = self.states.max(axis=0)
        print('hard coded min_max:')
        print(self.states_min_max_values)
        print('computed min, max:')
        print(states_computed_min)
        print(states_computed_max)

        for state_idx in range(self.states.shape[1]):
            assert self.states_min_max_values[state_idx, 0] < states_computed_min[state_idx]
            assert self.states_min_max_values[state_idx, 1] > states_computed_max[state_idx]
    
    def compute_probability(self, state):
        ## computing prob of x
        x_idx = np.searchsorted(self.states_bin_edges[0], state[0], side='right') - 1
        x_lb, x_ub = self.states_bin_edges[0][x_idx:x_idx+2] 
        prob_x = self.x_hist[x_idx] * self.states_resolution[0]
        print('prob x:', prob_x)


        ## computing prob of y given x
        x_states = self.states[:, 0]
        x_bounds = np.logical_and(x_states >= x_lb, x_states < x_ub)
        y_given_x_states = self.states[x_bounds][:, 1]
        y_give_x_hist, _ = np.histogram(y_given_x_states, self.states_bin_edges[1], density=True)

        y_idx = np.searchsorted(self.states_bin_edges[1], state[1], side='right') - 1
        prob_y_given_x = y_give_x_hist[y_idx] * self.states_resolution[1]
        print('prob y given x:', prob_y_given_x)
        

        ## computing prob of z given x and y
        y_lb, y_ub = self.states_bin_edges[1][y_idx:y_idx+2] 
        y_states = self.states[:, 1]
        y_bounds = np.logical_and(y_states >= y_lb, y_states < y_ub)
        xy_bounds = np.logical_and(x_bounds, y_bounds)
        z_given_xy_states = self.states[xy_bounds][:, 2]
        z_give_xy_hist, _ = np.histogram(z_given_xy_states, self.states_bin_edges[2], density=True)

        z_idx = np.searchsorted(self.states_bin_edges[2], state[2], side='right') - 1
        prob_z_given_xy = z_give_xy_hist[z_idx] * self.states_resolution[2]
        print('prob_z_given_xy', prob_z_given_xy)


        ## computing prob of yaw given x, y, and z
        z_lb, z_ub = self.states_bin_edges[2][z_idx:z_idx+2] 
        z_states = self.states[:, 2]
        z_bounds = np.logical_and(z_states >= z_lb, z_states < z_ub)
        xyz_bounds = np.logical_and(xy_bounds, z_bounds)
        yaw_given_xyz_states = self.states[xyz_bounds][:, 3]
        yaw_give_xyz_hist, _ = np.histogram(yaw_given_xyz_states, self.states_bin_edges[3], density=True)

        yaw_idx = np.searchsorted(self.states_bin_edges[3], state[3], side='right') - 1
        prob_yaw_given_xyz = yaw_give_xyz_hist[yaw_idx] * self.states_resolution[3]
        print('prob_yaw_given_xyz', prob_yaw_given_xyz)

        ## ploting
        # plt.figure()
        # plt.hist(self.states_bin_edges[0][:-1], self.states_bin_edges[0], weights=self.x_hist)
        # plt.figure()
        # plt.hist(self.states_bin_edges[1][:-1], self.states_bin_edges[1], weights=y_give_x_hist)
        # plt.figure()
        # plt.hist(self.states_bin_edges[2][:-1], self.states_bin_edges[2], weights=z_give_xy_hist)
        # plt.figure()
        # plt.hist(self.states_bin_edges[3][:-1], self.states_bin_edges[3], weights=yaw_give_xyz_hist)
        # plt.show()

        overall_prob = prob_x * prob_y_given_x * prob_z_given_xy * prob_yaw_given_xyz
        print('overall_prob', overall_prob)
        return overall_prob

    def compute_joint_prob(self, state):
        x_i = np.searchsorted(self.states_bin_edges[0], state[0], side='right') - 1
        y_i = np.searchsorted(self.states_bin_edges[1], state[1], side='right') - 1
        z_i = np.searchsorted(self.states_bin_edges[2], state[2], side='right') - 1
        yaw_i = np.searchsorted(self.states_bin_edges[3], state[3], side='right') - 1

        count = self.hist_dd[x_i, y_i, z_i, yaw_i]
        prob = count / self.states.shape[0]
        return prob
    
    def compute_states_prob(self):
        statesList = self.df['statesList'].tolist()
        states_prob_list = []
        for states_sequence in statesList:
            state = states_sequence[-1]
            if (state == None).all():
                prob = -1
            else:
                prob = self.compute_joint_prob(state)
            states_prob_list.append(prob)

        self.df['statesProbList'] = states_prob_list
        return self.df


def main():
    working_dir = '/home/majd/catkin_ws/src/basic_rl_agent/data2/flightgoggles/datasets/imageBezierData_I8_1000'
    allDataFileWithMarkersAndStates = 'allData_WITH_STATES_imageBezierData_I8_1000_20220418-1855.pkl'
    index = allDataFileWithMarkersAndStates.find('_WITH_STATES')
    assert index != -1
    index += len('_WITH_STATES')


    sp = StateProbability(os.path.join(working_dir, allDataFileWithMarkersAndStates))
    df_with_states_prob = sp.compute_states_prob()

    
    df_new_name = allDataFileWithMarkersAndStates[:index] + '_PROB' + allDataFileWithMarkersAndStates[index:]
    df_with_states_prob.to_pickle(os.path.join(working_dir, df_new_name))
    print('{} was saved.'.format(os.path.join(working_dir, df_new_name)))



if __name__ == '__main__':
    main()
    