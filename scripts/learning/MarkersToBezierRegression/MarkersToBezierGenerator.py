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

class MarkersAndTwistDataToBeizerDataGeneratorWithDataAugmentation(Sequence):
    def __init__(self, x_set, y_set, batch_size, inputImageShape, config, imageList=None):
        '''
            @param x_set: a list of two lists. The first list contains the markersData in an image. each markersData is an np array with shape=(4, 3).
                        The second list contains the twist data, a numpy array of shape (4, )
            @param y_set: a list of two lists that contains the control points of the position and yaw respectively.
            @param inputImageShape: the size of the images that markersData are stored for. Useful to compute the markersData ratio.
        '''
        self.markersDataSet = x_set[0]
        self.twistDataSet = x_set[1]

        self.positionControlPointsList, self.yawControlPointsList = y_set[0], y_set[1]
        self.batch_size = batch_size


        #compute markersDataFactor
        inputImageHeight, inputImageWidth, _ = inputImageShape
        self.markersDataFactor = np.array([1.0/float(inputImageWidth), 1.0/float(inputImageHeight), 1.0])

        self.config = config
        self.dataAugmentingRate = self.config['dataAugmentationRate']

        # markers data shape
        self.numOfImageSequence = config['numOfImageSequence']
        self.markersNetworkType = config['markersNetworkType'] # Dense, LSTM
        if self.markersNetworkType == 'Dense':
            self.markersDataShape = (12 * self.numOfImageSequence, )
        elif self.markersNetworkType == 'LSTM':
            self.markersDataShape = (self.numOfImageSequence, 12)
        else:
            raise NotImplementedError

        # twist data function selection
        twistDataGenType = self.config['twistDataGenType']
        if twistDataGenType == 'last2points_and_EMA':
            self.alpha = self.config['alpha']
            self.twistDataGenFunction = self.__genTwistData_3point_last2_and_average
        elif twistDataGenType == 'EMA':
            self.alpha = self.config['alpha']
            self.twistDataGenFunction = self.__genTwistData_eponentialMovingAverage
        elif twistDataGenType == 'last2points':
            self.twistDataGenFunction = self.__genTwistData_last2points
        elif twistDataGenType == 'Sequence':
            self.numOfTwistSequence = self.config['numOfTwistSequence']
            self.twistDataGenFunction = self.__genTwistData_Sequence

            # determine the twistDataTargetReturn
            twistNetworkType = self.config['twistNetworkType']
            if twistNetworkType == 'LSTM':
                self.twistDataTargetShape = (self.numOfTwistSequence, 4)
            elif twistNetworkType == 'Dense':
                self.twistDataTargetShape = (self.numOfTwistSequence*4, )
            else:
                raise NotImplementedError
            
        else:
            raise NotImplementedError

        if self.dataAugmentingRate != 0:
            if self.numOfImageSequence == 1:
                self.dataAugmentiationEnabled = True
            else:
                raise NotImplementedError
        else:
            self.dataAugmentiationEnabled = False

        self.imageList = imageList

        # data cleaning
        self.__removeZerosMarkersAndNanTwistData()

    def __removeZerosMarkersAndNanTwistData(self):
        remove_indices = []
        for idx, markersData in enumerate(self.markersDataSet):
            if (markersData[:, :, -1] == 0).any() or np.isnan(self.twistDataSet[idx]).any(): # check if the Z component of any marker is zeros or if any twist value is nan
                remove_indices.append(idx)
        markersDataTmpList = []
        twistDataTmpList = []
        positionCpList = []
        yawCpList = []
        imageList = [] if not self.imageList is None else None
        for i in range(len(self.markersDataSet)):
            if not i in remove_indices:
                markersDataTmpList.append(self.markersDataSet[i])
                twistDataTmpList.append(self.twistDataSet[i])
                positionCpList.append(self.positionControlPointsList[i])
                yawCpList.append(self.yawControlPointsList[i])

                if not self.imageList is None:
                    imageList.append(self.imageList[i])
        self.markersDataSet = markersDataTmpList            
        self.twistDataSet = twistDataTmpList
        self.positionControlPointsList = positionCpList
        self.yawControlPointsList = yawCpList
        self.imageList = imageList

    def getImageList(self):
        return self.imageList

    def getIndex(self, index):
        '''
            for debugging only
        '''
        x = range(min(self.batch_size, len(self.markersDataSet)-index*self.batch_size))
        return (x, self.batch_size)
    
    def __len__(self):
        return math.ceil(len(self.markersDataSet) / self.batch_size)

    def __getitem__(self, index):
        '''
            Generates data containing batch_size samples with data augmentation.
            the data augmentation is appled to the markersData, where one of them is randomly set to zero (the location of the marker and its depth).
            @return a list: 1. a list og lists: 1. markersData_batch 2. twistData_batch for x.
                            2. a list of two lists: 1. positionControlPoints_batch, 2. yawControlPoints_batch for y.
        '''
        markersData_batch = []
        twistData_batch = []
        positionControlPoints_batch = []
        yawControlPoints_batch = []

        for row in range(min(self.batch_size, len(self.markersDataSet)-index*self.batch_size)):
            markersData = self.markersDataSet[index*self.batch_size + row]
            assert (markersData[:, -1] != 0).any(), 'markersData have Z component euqals to zero' # check if the Z component of any marker is zeros.
            # normailze markersData:
            markersDataNormalized = np.multiply(markersData, self.markersDataFactor)

            # apply data augmentation before reshaping:
            # TODO: data augmentation for sequence of images
            if self.dataAugmentiationEnabled:
                if np.random.rand() <= self.dataAugmentingRate:
                    randomIdx = np.random.randint(0, 4)
                    markersDataNormalized[randomIdx] = np.array([0, 0, 0])

            markersDataNormalized = markersDataNormalized[:self.numOfImageSequence].reshape(self.markersDataShape)

            twistData = self.twistDataSet[index*self.batch_size + row]

            twistData = self.twistDataGenFunction(twistData)

            positionControlPoints = self.positionControlPointsList[index*self.batch_size + row]
            positionControlPoints = np.array(positionControlPoints).reshape(15, )

            yawControlPoints = self.yawControlPointsList[index*self.batch_size + row]
            yawControlPoints = np.array(yawControlPoints)

            markersData_batch.append(markersDataNormalized)
            twistData_batch.append(twistData)
            positionControlPoints_batch.append(positionControlPoints)
            yawControlPoints_batch.append(yawControlPoints)

        markersData_batch = np.array(markersData_batch)
        twistData_batch = np.array(twistData_batch)
        positionControlPoints_batch = np.array(positionControlPoints_batch)
        yawControlPoints_batch = np.array(yawControlPoints_batch)
        return ([markersData_batch, twistData_batch], [positionControlPoints_batch, yawControlPoints_batch])

    def __genTwistData_last2points(self, twistData):
        '''
            @returns the last two points
        '''
        return np.concatenate([twistData[-1], twistData[-2]], axis=0) 

    def __genTwistData_3point_last2_and_average(self, twistData):
        '''
            @returns 3 twist data points: the last 2 points and the exponential moving average (EMA) 
        '''
        averageTwist = twistData[0]
        for currTwist in twistData[1:]:
            averageTwist = self.alpha * currTwist + (1-self.alpha) * averageTwist
        return np.concatenate([twistData[-1], twistData[-2], averageTwist], axis=0) 

    def __genTwistData_eponentialMovingAverage(self, twistData):
        '''
            @returns the exponential moving average (EMA) of the twistData
        '''
        averageTwist = twistData[0]
        for currTwist in twistData[1:]:
            averageTwist = self.alpha * currTwist + (1-self.alpha) * averageTwist
        return averageTwist

    def __genTwistData_Sequence(self, twistData):
        return twistData[-self.numOfTwistSequence:].reshape(self.twistDataTargetShape)

# end of class

def test_MarkersAndTwistDataToBezierDataGeneratorWithDataAugmentation(directory):
    # configuration file:
    config = {
        'alpha': 0.1,
        'dataAugmentationRate': 0.0,
        'numOfImageSequence': 3,
        'markersNetworkType': 'LSTM',  # 'Dense'
        'twistNetworkType': 'LSTM',
        'twistDataGenType': 'Sequence',
        'numOfTwistSequence': 40
    }

    df = pd.read_pickle(directory)
    df = df.sample(frac=0.5)
    Xset = [df['markersData'].tolist(), df['vel'].tolist()]
    Yset = [df['positionControlPoints'].tolist(), df['yawControlPoints'].tolist()]
    imageList = df['images'].tolist()
    batchSize = 1
    inputImageShape=(480, 640, 3)
    markersDataReverseFactor = np.array([inputImageShape[1], inputImageShape[0], 1], dtype=np.float32)
    dataGen = MarkersAndTwistDataToBeizerDataGeneratorWithDataAugmentation(Xset, Yset, batchSize, inputImageShape, config, imageList)
    imageList = dataGen.getImageList()

    numOfImageSeq = config['numOfImageSequence']
    assert numOfImageSeq <= imageList[0].shape[0]

    bezierVisulizer = BezierVisulizer(plot_delay=1, numOfImageSequence=numOfImageSeq)

    for i in range(dataGen.__len__()):
        Xbatch, Ybatch = dataGen.__getitem__(i)
        print('working on batch #{}'.format(i))
        x_range, bs = dataGen.getIndex(i)
        imageBatch = [imageList[i*bs+k] for k in x_range]
        for idx, imageNameSeq in enumerate(imageBatch):
            print(imageNameSeq.shape)
            # print('xbatch.shape:', Xbatch[0].shape, Xbatch[1].shape)

            # twist data:
            twistData = Xbatch[1][idx]
            print('twistData:')
            print(twistData.reshape(-1, 4).shape)
            print()

            # markers data:
            markersDataNormalized = Xbatch[0][idx]
            markersDataNormalized = markersDataNormalized.reshape(numOfImageSeq, 4, 3)

            # testing markersData data augmentation:
            if (markersDataNormalized == 0).any():
                print('markersData has zeros', markersDataNormalized)

            markersData = np.multiply(markersDataNormalized, markersDataReverseFactor)

            imageSeq = []
            for seqId in range(numOfImageSeq):
                image = cv2.imread(imageNameSeq[seqId,  0])
                for marker in markersData[seqId, :, :-1]:
                    marker = marker.astype(np.int)
                    image = cv2.circle(image, (marker[0], marker[1]), radius=6, color=(255, 0, 0), thickness=-1)
                imageSeq.append(image)
            positonCP, yawCP = Ybatch[0][idx], Ybatch[1][idx]
            print('Ybatch.shape:', positonCP.shape, yawCP.shape)
            positonCP = positonCP.reshape(5, 3).T
            yawCP = yawCP.reshape(1, 3)
            bezierVisulizer.plotBezier(imageSeq, positonCP, yawCP)

    
def main():
    allDataFileWithMarkers = '/home/majd/catkin_ws/src/basic_rl_agent/data/imageBezierData1/allDataWithMarkers.pkl'
    test_MarkersAndTwistDataToBezierDataGeneratorWithDataAugmentation(allDataFileWithMarkers)
   

if __name__ == '__main__':
    main()