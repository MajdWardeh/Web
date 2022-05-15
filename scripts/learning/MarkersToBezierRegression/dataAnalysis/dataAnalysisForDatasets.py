import sys

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
import math
import cv2
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Input, layers, Model, backend as k
from tensorflow.keras.utils import Sequence
from learning.markersDataUtils.imageMarkers_dataPreprocessing import ImageMarkersGroundTruthPreprocessing
from Bezier_untils import BezierVisulizer

def dataVisulizer(directory):
    df = pd.read_pickle(directory)
    df = df.sample(frac=0.001)
    imagesList = df['images'].tolist()
    markersList = df['markersData'].tolist()
    positionCP_list = df['positionControlPoints'].tolist()
    yawCP_list = df['yawControlPoints'].tolist()

    numOfSeq_image = imagesList[0].shape[0]
    numOfChannels_image = imagesList[0].shape[1]

    bezierVisulizer = BezierVisulizer(plot_delay=1, numOfImageSequence=numOfSeq_image)

    assert numOfChannels_image == 1
    for i in range(len(imagesList)):
        # collect images to plot
        imagesToPlot = []
        images_np = imagesList[i]
        markers_seq = markersList[i]
        for seqId in range(numOfSeq_image):
            imageName = images_np[seqId, 0]
            image = cv2.imread(imageName)
            markers_image = markers_seq[seqId]
            for m in markers_image:
                m = m.astype(np.int)
                c = (m[0], m[1])
                image = cv2.circle(image, c, 10, (0, 255, 0), thickness=-1)
            imagesToPlot.append(image)
        bezierVisulizer.plotBezier(imagesToPlot, np.array(positionCP_list[i]).T, np.array(yawCP_list[i]).reshape(3, 1).T)
    
def statisticalAnalysisForImageSequenceTime(directory):
    df = pd.read_pickle(directory)
    # df = df.sample(frac=1)
    imagesList = df['images'].tolist()
    markersList = df['markersData'].tolist()
    positionCP_list = df['positionControlPoints'].tolist()
    yawCP_list = df['yawControlPoints'].tolist()

    numOfSeq_image = imagesList[0].shape[0]
    numOfChannels_image = imagesList[0].shape[1]
    assert numOfChannels_image == 1

    AllStampDiffs = []
    for i in range(len(imagesList[:])):
        # collect images to plot
        imagesToPlot = []
        images_np = imagesList[i]
        markers_seq = markersList[i]

        imageStamps = []
        for seqId in range(numOfSeq_image):
            imageName = images_np[seqId, 0].split('/')[-1]
            imageName = imageName.split('_')[-1]
            imageName = imageName.split('.')[0]
            stamp = int(imageName)
            imageStamps.append(stamp)

        stampDiff = []
        for k in range(len(imageStamps)-1):
            stampDiff.append(imageStamps[k+1]-imageStamps[k])
        AllStampDiffs.append(stampDiff)
    
    AllStampDiffs = np.array(AllStampDiffs)
    print('mean:', np.mean(AllStampDiffs, axis=0))
    print('std:', np.std(AllStampDiffs, axis=0))
    print('max:', np.max(AllStampDiffs, axis=0))
    print('min:', np.min(AllStampDiffs, axis=0))



def plot_the_starting_position(directory):
    df = pd.read_pickle(directory)
    # df = df.sample(frac=1)
    imagesList = df['images'].tolist()
    markersList = df['markersData'].tolist()
    positionCP_list = df['positionControlPoints'].tolist()
    yawCP_list = df['yawControlPoints'].tolist()
    diff_list = []
    for markers in markersList:
        error = markers - markers[0]
        mse = np.square(error).mean()
        diff_list.append(mse)
    diff_list = np.array(diff_list)
    
    print(diff_list.min(), diff_list.max())
    print(diff_list[diff_list == 0.0].shape)

def normalize_markers_and_twist_data(directory):
    df = pd.read_pickle(directory)
    markersList = np.array(df['markersData'].tolist())
    twistDataList = np.array(df['vel'].tolist())
    statesProbList = np.array(df['statesProbList'].tolist())

    indices_filtered = []
    for idx, markersData in enumerate(markersList):
        if (markersData[:, :, -1] != 0).all() and not np.isnan(twistDataList[idx]).all() and statesProbList[idx] != -1:
            indices_filtered.append(idx)
    print('markersList shape:', markersList.shape)
    markersList_filtered = markersList[indices_filtered]
    twistDataList_filtered = twistDataList[indices_filtered]
    statesProbList_filtered = statesProbList[indices_filtered]
    print('markersList_filtered shape:', markersList_filtered.shape)

    markersList_filtered_reshaped = markersList_filtered.reshape(-1, 3)
    markers_min = markersList_filtered_reshaped.min(axis=0)
    markers_max = markersList_filtered_reshaped.max(axis=0)
    print('markersList filtered min: {}, max: {}'.format(markers_min, markers_max))
    markersData_hardcoded_min_max = np.array([
        (0, 640),
        (0, 480),
        (1, 25)
    ],dtype=np.float32)
    print('hardcoded markers min_max:')
    print(markersData_hardcoded_min_max)

    markersData_hardcoded_mean = np.array([322.00710606, 176.19885206, 12.77271492])
    markersData_hardcoded_std = np.array([124.70433658, 73.10797561, 5.49590978])

    print('markersList computed mean: {}, std: {}'.format(markersList_filtered_reshaped.mean(axis=0), markersList_filtered_reshaped.std(axis=0)))


    markersList_filtered_reshaped_normalized = \
        (markersList_filtered_reshaped - markersData_hardcoded_mean) / markersData_hardcoded_std
    print('markersList filtered, normalized:')
    print('min: {}, max: {}'.format(markersList_filtered_reshaped_normalized.min(axis=0), markersList_filtered_reshaped_normalized.max(axis=0)))
    print('std: ', markersList_filtered_reshaped_normalized.std(axis=0))
    markersList_filtered_normalized = markersList_filtered_reshaped_normalized.reshape(markersList_filtered.shape)

    print('-------------------------------------')

    ## normalize twistData:
    print('normalizing twistData:')
    twistData_min = twistDataList_filtered.reshape(-1, 4).min(axis=0)
    twistData_max = twistDataList_filtered.reshape(-1, 4).max(axis=0)
    print('twisData: min: {}, max: {}'.format(twistData_min, twistData_max))

    twistData_hardcoded_mean = np.array([3.5, 0., 1.25, 0.])
    twistData_hardcoded_std = np.array([2.39911219, 1.36576634, 0.68722698, 0.10353576]) 
    print('twist Data hardcoded mean: {}, std: {}'.format(twistData_hardcoded_mean, twistData_hardcoded_std))

    twistDataList_filtered_reshaped = twistDataList_filtered.reshape(-1, 4)
    twistData_computed_mean = twistDataList_filtered_reshaped.mean(axis=0)
    twistData_computed_std = twistDataList_filtered_reshaped.std(axis=0)

    print('twist Data computed mean: {}, std: {}'.format(twistData_computed_mean, twistData_computed_std))

    twistDataList_filtered_reshaped_normalized = \
        (twistDataList_filtered_reshaped - twistData_hardcoded_mean) / twistData_hardcoded_std
    # twistDataList_filtered_reshaped_normalized = \
    #     (twistDataList_filtered_reshaped - twistData_hard_coded_min_max[:, 0]) / (twistData_hard_coded_min_max[:, 1] - twistData_hard_coded_min_max[:, 0])
    # twistDataList_filtered_reshaped_normalized = 2*twistDataList_filtered_reshaped_normalized - 1
    print('twistData filtered, normalized:')
    print('min: {}, max: {}'.format(twistDataList_filtered_reshaped_normalized.min(axis=0), twistDataList_filtered_reshaped_normalized.max(axis=0)))
    print('std:', twistDataList_filtered_reshaped_normalized.std(axis=0))
    twistDataList_filtered_normalized = twistDataList_filtered_reshaped_normalized.reshape(twistDataList_filtered.shape)

def normalize_controlPoints(directory):
    df = pd.read_pickle(directory)
    print(df.columns)
    positionCPList = np.array(df['positionControlPoints'].tolist())
    yawCPList = np.array(df['yawControlPoints'].tolist())
    print(positionCPList.shape)
    print(yawCPList.shape)

    print('position:')
    positionCP_mean = positionCPList.mean(axis=0)
    positionCP_std = positionCPList.std(axis=0)
    print('mean shape', positionCP_mean.shape)
    print('mean:', positionCP_mean)
    print('std:', positionCP_std)

    print('yaw:')
    print('mean:', yawCPList.mean(axis=0))
    print('std:', yawCPList.std(axis=0))



    
def main():
    allDataFileWithMarkers = '/home/majd/catkin_ws/src/basic_rl_agent/data2/flightgoggles/datasets/imageBezierData_I8_1000/allData_WITH_STATES_imageBezierData_I8_1000_20220418-1855.pkl'
    # dataVisulizer(allDataFileWithMarkers)
    # statisticalAnalysisForImageSequenceTime(allDataFileWithMarkers)
    # plot_the_starting_position(allDataFileWithMarkers)

    # normalize_markers_and_twist_data(allDataFileWithMarkers)
    normalize_controlPoints(allDataFileWithMarkers)

   

if __name__ == '__main__':
    main()