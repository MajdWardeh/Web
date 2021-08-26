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

    
def main():
    allDataFileWithMarkers = '/home/majd/catkin_ws/src/basic_rl_agent/data/imageBezierData1/allDataWithMarkers.pkl'
    # dataVisulizer(allDataFileWithMarkers)
    statisticalAnalysisForImageSequenceTime(allDataFileWithMarkers)

   

if __name__ == '__main__':
    main()