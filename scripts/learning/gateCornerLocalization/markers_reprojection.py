import os
import numpy as np
import numpy.linalg as la
from scipy.spatial.transform import Rotation as R


import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2



def camera_info():
    K = np.array([342.7555236816406, 0.0, 320.0, 0.0,
                342.7555236816406, 240.0, 0.0, 0.0, 1.0]).reshape(3, 3)
    return K

def plotMarkers(image, markers, color=(0, 0, 255)):
    '''
        @arg markers: np array of shape=(rows, columns), where min(rows)=1, min(columns)=2
    ''' 
    markers = markers[:, :2]  # remove the z component.
    markers = markers.astype(np.int)
    for i, marker in enumerate(markers):
        image = cv2.circle(image, tuple(marker), 5, color, thickness=-1)
        cv2.putText(image, str(i), tuple(marker), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    return image

def compute_markers_3d_position(markers, fx, fy, cx, cy):
    '''
        @arg markers: np array of shape (4, 3). each row of markers has u, v, Z where
        u and v are the locations of the markers in the image, and Z is their depth in meters.
        @arg fx, fy: are the focal length of the camera calibration matrix
        @arg cx, cy: are the prenciple point of the camera calibration matrix

        return markers_3d of shape (3, 4), where colum c is the 3d location of the cth marker in markers in camera frame.
    '''
    assert markers.shape == (4, 3), 'markers shape must be (4, 3)'
    assert np.sum(markers[:, -1] != 0, axis=0)==4, 'all markers must be available'
    markers_u = markers[:, 0]
    markers_v = markers[:, 1]
    markers_Z = markers[:, 2]
    markers_3d = np.zeros_like(markers)
    markers_3d[:, 2] = markers_Z
    for i in range(markers.shape[0]):
        markers_3d[i, 0] = (markers_Z[i]/fx) * (markers_u[i] - cx)
        markers_3d[i, 1] = (markers_Z[i]/fy) * (markers_v[i] - cy)

    return markers_3d.T

def compute_point_projection_on_image(p, K):
    '''
        @arg p: np vector of shape (3, ) or (3, 1)
        @arg K: the intrinsic camera matrix 
    '''
    p = p.reshape(3, 1)
    p_image = np.matmul(K, p)
    p_image /= p_image[2] # divide over the z component
    return p_image[:2]

def compute_spherical_coordinates(p):
    '''
        @arg p: np array of shape (3, ) or (3, 1)
        @return np array, the spherical coordinates of the vector p 
    '''
    x = p[0]
    y = p[1]
    z = p[2]

    r = la.norm(p)
    theta = np.arccos(z/r)
    psi = np.arctan(y/x) 

    return np.array([r, theta, psi])

def compute_gate_heading(markers_3d, degrees=False):
    '''
        computers the heading (yaw angle) of a gate given its markers in 3d.
        @arg markers_3d: np array of shape (3, 4) the 3d location of the markers in the camera frame; the Z axis represents the depth
    '''
    assert markers_3d.shape == (3, 4), 'Error, markers shape is not as expected'
    assert (markers_3d[2, :] >= 0).any(), 'Error, one or more of markers_3d.Z are negative. \
                Are you sure that markers_3d are represented in the camera frame?'

    if markers_3d[0, 0] > markers_3d[0, 1]:
        right_idx = 0 
        left_idx = 1
    else:
        right_idx = 1 
        left_idx = 0

    # compute the averge Z component of the left and right sides 
    # the IDs of the left side markers are 1 and 3 (the odds)
    left_side_averge_Z = (markers_3d[2, left_idx] + markers_3d[2, left_idx+2]) / 2.0
    right_side_averge_Z = (markers_3d[2, right_idx] + markers_3d[2, right_idx+2]) / 2.0
    right_minus_left_Z = right_side_averge_Z - left_side_averge_Z

    # the same of the X component
    left_side_averge_X = (markers_3d[0, left_idx] + markers_3d[0, left_idx+2]) / 2.0
    right_side_averge_X = (markers_3d[0, right_idx] + markers_3d[0, right_idx+2]) / 2.0
    right_minus_left_X = right_side_averge_X - left_side_averge_X
    
    distance_Z = right_minus_left_Z / 2
    distance_X = right_minus_left_X / 2

    # we want the angle of arctan(z/x) 
    yaw = np.arctan2(distance_Z, distance_X)
    if degrees:
        yaw = yaw * 180.0/np.pi
    return yaw 


def extend_df(df):
    K = camera_info()
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    markers_list = df['markersArrays'].tolist()
    markers_3d_list = []
    heading_list = []
    gate_center_spherical_list = []
    for markers in markers_list:
        markers_3d = compute_markers_3d_position(markers, fx, fy, cx, cy)
        heading = compute_gate_heading(markers_3d)
        gate_center_3d = markers_3d.mean(axis=1)
        gate_center_spherical = compute_spherical_coordinates(gate_center_3d)

        assert abs(np.degrees(heading)) < 90, 'Error, the abs value of the heading must be (strictly) less than 90'

        markers_3d_list.append(markers_3d)
        heading_list.append(heading)
        gate_center_spherical_list.append(gate_center_spherical)
    df['markers_3d'] = markers_3d_list
    df['gateHeading'] = heading_list
    df['gateCenterSpherical'] = gate_center_spherical_list
    return df

def testing():

    from imageMarkersDataSaverLoader import ImageMarkersDataLoader
    from imageMarkersDatasetsMerging import mergeDatasets
    base_path = '/home/majd/catkin_ws/src/basic_rl_agent/data/imageMarkersAnglesTest'
    dataframe = mergeDatasets(base_path)
    print(dataframe.head())

    images_list = dataframe['images'].tolist()
    markers_list = dataframe['markersArrays'].tolist()

    K = camera_info()
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    for i in range(len(images_list)):
        markers = markers_list[i]

        image_path = images_list[i]

        gate_image = cv2.imread(image_path)

        markers_3d = compute_markers_3d_position(markers, fx, fy, cx, cy)
        gate_center_3d = markers_3d.mean(axis=1)
        gate_center_projection = compute_point_projection_on_image(gate_center_3d, K)

        gate_marker_image = plotMarkers(gate_image, markers)
        gate_marker_center_image = plotMarkers(gate_marker_image, gate_center_projection.reshape(1, 2))

        gate_yaw = compute_gate_heading(markers_3d)
        print(gate_yaw*180/np.pi)

        cv2.imshow('image', gate_image)
        cv2.imshow('image2', gate_marker_center_image)
        if cv2.waitKey(0) == ord('q'):
            break


if __name__ == '__main__':
    testing()
