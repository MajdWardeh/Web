import os
import numpy as np
import numpy.linalg as la
from scipy.spatial.transform import Rotation as R


import sys
sys.path.append('/home/majd/catkin_ws/src/basic_rl_agent/scripts/learning')
sys.path.append(
    '/home/majd/catkin_ws/src/basic_rl_agent/scripts/learning/gateCornerLocalization')
sys.path.append('/home/majd/catkin_ws/src/basic_rl_agent/scripts/learning/end_to_end_learning')

ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2

from gateCornerLocalization.imageMarkersDataSaverLoader import ImageMarkersDataLoader
from gateCornerLocalization.imageMarkersDatasetsMerging import mergeDatasets, remove_samples_with_less_markers
from gateCornerLocalization.markers_reprojection import extend_df, plotMarkers

def plot_gate(image_path, markers, gate_yaw):
    gate_image = cv2.imread(image_path)

    plotMarkers(gate_image, markers)
    gate_yaw = gate_yaw * 180./np.pi
    cv2.putText(gate_image, str(round(gate_yaw, 3)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('image', gate_image)

def compute_gate_min_max_data(gate_center_spherical_list, gate_heading_list):
    print(gate_center_spherical_list.shape)
    print(gate_heading_list.shape)
    r_min = gate_center_spherical_list[:, 0].min()
    r_max = gate_center_spherical_list[:, 0].max()
    theta_min = gate_center_spherical_list[:, 1].min()
    theta_max = gate_center_spherical_list[:, 1].max()
    phi_min = gate_center_spherical_list[:, 2].min()
    phi_max = gate_center_spherical_list[:, 2].max()
    yaw_min = gate_heading_list.min()
    yaw_max = gate_heading_list.max()
    return (r_min, r_max), (theta_min, theta_max), (phi_min, phi_max), (yaw_min, yaw_max)


def test():
    base_path = '/home/majd/catkin_ws/src/basic_rl_agent/data/imageMarkersDataWithDronePoses'
    df = mergeDatasets(base_path, drop_dronePoses=True, drop_gatePoses=True)
    df = remove_samples_with_less_markers(df)
    df = extend_df(df)
    images_list = np.array(df['images'].tolist())
    markers_list = df['markersArrays'].tolist()
    markers_3d_list = df['markers_3d'].tolist()
    gate_heading_list = np.array(df['gateHeading'].tolist())
    gate_center_spherical_list = np.array(df['gateCenterSpherical'].tolist())

    r_range, theta_range, phi_range, yaw_range = compute_gate_min_max_data(gate_center_spherical_list, gate_heading_list)
    print('r_range', r_range)
    print('theta_range', theta_range)
    # print(np.array(yaw_range)*180./np.pi)
    print('phi_range, in degrees', np.degrees(phi_range))
    print(np.array(yaw_range)*180./np.pi)
    print('yaw mean: {}'.format(gate_heading_list.mean()))
    print('yaw std in degrees: {}'.format(gate_heading_list.std()*180./np.pi))



if __name__ == '__main__':
    test()
