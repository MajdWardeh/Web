import sys
import gc
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
import math
import cv2
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Input, layers, Model, backend as k
from tensorflow.keras.utils import Sequence
from learning.markersDataUtils.imageMarkers_dataPreprocessing import ImageMarkersGroundTruthPreprocessing
from Bezier_untils import BezierVisulizer

class MarkersAndTwistDataToBeizerDataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size, inputImageShape, config, imageList=None, statesProbList=None, normalizationType='new'):
        '''
            @param x_set: a list of two lists. The first list contains the markersData in an image. each markersData is an np array with shape=(4, 3).
                        The second list contains the twist data, a numpy array of shape (4, )
            @param y_set: a list of two lists that contains the control points of the position and yaw respectively.
            @param inputImageShape: the size of the images that markersData are stored for. Useful to compute the markersData ratio.
        '''
        self.markersDataSet = np.array(x_set[0], dtype=np.float32)
        self.twistDataSet = np.array(x_set[1], dtype=np.float32)

        self.positionControlPointsList, self.yawControlPointsList = y_set[0], y_set[1]
        self.batch_size = batch_size


        #compute markersDataFactor
        inputImageHeight, inputImageWidth, _ = inputImageShape
        self.markersDataFactor = np.array([1.0/float(inputImageWidth), 1.0/float(inputImageHeight), 1.0])

        self.process_config(config)


        if self.dataAugmentingRate != 0:
            if self.numOfImageSequence == 1:
                self.dataAugmentiationEnabled = True
            else:
                raise NotImplementedError
        else:
            self.dataAugmentiationEnabled = False

        self.imageList = imageList
        self.sampleWeightList = 1. - np.array(statesProbList, dtype=np.float32) if statesProbList is not None else None
        self.sampleWeightEnabled = statesProbList is not None

        # data cleaning
        
        # self.__removeZerosMarkersAndNanTwistData()
        warnings.warn('removing the zero Markers and NanTwistData is NOT called')

        print('normalization type: {}'.format(normalizationType))

        if normalizationType == 'new':
            self.__normalizeMarkersAndTwistData()
        elif normalizationType == 'old':
            pass
        else: 
            raise RuntimeError('only "new" or "old" values are accepted for normiazationType')
        self.normalizationType = normalizationType 

        # save_path = '/home/majd/catkin_ws/src/basic_rl_agent/data2/flightgoggles/datasets/imageBezierDataV2_1_1000/allData_WITH_STATES_PROB_filtered_imageBezierDataV2_1_1000_20220416-1501.pkl' 
        # data = {
        #     'vel': self.twistDataSet,
        #     'markersData': self.markersDataSet,
        #     'positionControlPoints': self.positionControlPointsList,
        #     'yawControlPoints': self.yawControlPointsList,
        #     'statesProbList': np.array([1. - s for s in self.sampleWeightList], dtype=np.float32).tolist()
        # }
        # df = pd.DataFrame(data)
        # print(df.columns)
        # df.to_pickle(save_path)
        # print('df saved')

        ## call Garbage Collector
        gc.collect()
    
    def process_config(self, config):
        self.config = config

        self.markersData_hardcoded_mean = self.config.get('markersData_hardcoded_mean', \
                    np.array([322.00710606, 176.19885206, 12.77271492]))
        self.markersData_hardcoded_std = self.config.get('markersData_hardcoded_std', \
                    np.array([124.70433658, 73.10797561, 5.49590978]))

        self.twistData_hardcoded_mean = self.config.get('twistData_hardcoded_mean', \
                    np.array([3.5, 0., 1.25, 0.]))
        self.twistData_hardcoded_std = self.config.get('twistData_hardcoded_std', \
                    np.array([2.39911219, 1.36576634, 0.68722698, 0.10353576]))

        self.dataAugmentingRate = self.config['dataAugmentationRate']

        # markers data shape
        self.numOfImageSequence = config['numOfImageSequence']
        self.markersNetworkType = config['markersNetworkType'] # Dense, LSTM
        if self.markersNetworkType == 'Dense' or self.markersNetworkType == 'Separate_Dense':
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
            if twistNetworkType == 'LSTM' or twistNetworkType == 'Conv':
                self.twistDataTargetShape = (self.numOfTwistSequence, 4)
            elif twistNetworkType == 'Dense' or twistNetworkType == 'Separate_Dense':
                self.twistDataTargetShape = (self.numOfTwistSequence*4, )
            else:
                raise NotImplementedError

        else:
            raise NotImplementedError

    def __removeZerosMarkersAndNanTwistData(self):
        remove_indices = []
        for idx, markersData in enumerate(self.markersDataSet):
            if (markersData[:, :, -1] == 0).any() or np.isnan(self.twistDataSet[idx]).any(): # check if the Z component of any marker is zeros or if any twist value is nan
                remove_indices.append(idx)
        markersDataTmpList = []
        twistDataTmpList = []
        positionCpList = []
        yawCpList = []
        imageList = [] if self.imageList is not None else None
        sampleWeightList = [] if self.sampleWeightList is not None else None
        for i in range(len(self.markersDataSet)):
            if not i in remove_indices:
                markersDataTmpList.append(self.markersDataSet[i])
                twistDataTmpList.append(self.twistDataSet[i])
                positionCpList.append(self.positionControlPointsList[i])
                yawCpList.append(self.yawControlPointsList[i])

                if self.imageList is not None:
                    imageList.append(self.imageList[i])

                if self.sampleWeightList is not None:
                    sampleWeightList.append(self.sampleWeightList[i])

        self.markersDataSet = markersDataTmpList            
        self.twistDataSet = twistDataTmpList
        self.positionControlPointsList = positionCpList
        self.yawControlPointsList = yawCpList
        self.imageList = imageList
        self.sampleWeightList = sampleWeightList

    def __normalizeMarkersAndTwistData(self):
        ## normalizing markersDataSet
        markerData_np = np.array(self.markersDataSet)
        markersData_np_reshaped = markerData_np.reshape(-1, 3)
        print('markersData statistics:')
        print('Before normalizing:')
        print('min: ', markersData_np_reshaped.min(axis=0))
        print('max: ', markersData_np_reshaped.max(axis=0))
        print('mean: ', markersData_np_reshaped.mean(axis=0))
        print('std: ', markersData_np_reshaped.std(axis=0))

        markersData_np_reshaped_normalized = (markersData_np_reshaped - self.markersData_hardcoded_mean) / self.markersData_hardcoded_std
        print('After normalizing:')
        print('min: ', markersData_np_reshaped_normalized.min(axis=0))
        print('max: ', markersData_np_reshaped_normalized.max(axis=0))
        print('mean: ', markersData_np_reshaped_normalized.mean(axis=0))
        print('std: ', markersData_np_reshaped_normalized.std(axis=0))
        self.markersDataSet = markersData_np_reshaped_normalized.reshape(markerData_np.shape).tolist()
        print('-------------------------------------')

        ## normalizing twistDataSet
        twistData_np = np.array(self.twistDataSet)
        twistData_np_reshaped = twistData_np.reshape(-1, 4)
        print('twistData statistics:')
        print('Before normalizing:')
        print('min: ', twistData_np_reshaped.min(axis=0))
        print('max: ', twistData_np_reshaped.max(axis=0))
        print('mean: ', twistData_np_reshaped.mean(axis=0))
        print('std: ', twistData_np_reshaped.std(axis=0))
        twistData_np_reshaped_normalized = (twistData_np_reshaped - self.twistData_hardcoded_mean) / self.twistData_hardcoded_std
        print('After normalizing:')
        print('min: ', twistData_np_reshaped_normalized.min(axis=0))
        print('max: ', twistData_np_reshaped_normalized.max(axis=0))
        print('mean: ', twistData_np_reshaped_normalized.mean(axis=0))
        print('std: ', twistData_np_reshaped_normalized.std(axis=0))
        self.twistDataSet = twistData_np_reshaped_normalized.reshape(twistData_np.shape).tolist()
        print('-------------------------------------')


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

        if self.sampleWeightEnabled:
            sampleWeight_batch = []

        for row in range(min(self.batch_size, len(self.markersDataSet)-index*self.batch_size)):
            markersData = self.markersDataSet[index*self.batch_size + row]

            if self.normalizationType == 'old':
                # assert (markersData[:, -1] != 0).any(), 'markersData have Z component euqals to zero' # check if the Z component of any marker is zeros.
                # normailze markersData:
                markersDataNormalized = np.multiply(markersData, self.markersDataFactor)
            elif self.normalizationType == 'new':
                markersDataNormalized = np.array(markersData)

            # apply data augmentation before reshaping:
            # TODO: data augmentation for sequence of images
            if self.dataAugmentiationEnabled:
                if np.random.rand() <= self.dataAugmentingRate:
                    randomIdx = np.random.randint(0, 4)
                    markersDataNormalized[randomIdx] = np.array([0, 0, 0])

            markersDataNormalized = markersDataNormalized[:self.numOfImageSequence].reshape(self.markersDataShape)

            twistData = self.twistDataSet[index*self.batch_size + row]

            # twistData_normalized = np.multiply(twistData, 1./self.maxTwistValues)
            twistData_normalized = np.array(twistData)

            twistData_normalized = self.twistDataGenFunction(twistData_normalized)

            if self.sampleWeightEnabled:
                sampleWeight_batch.append(self.sampleWeightList[index*self.batch_size + row])

            positionControlPoints = self.positionControlPointsList[index*self.batch_size + row]
            # print('positionControlPoints shape', np.array(positionControlPoints).shape)
            positionControlPoints = np.array(positionControlPoints).reshape(15, )

            yawControlPoints = self.yawControlPointsList[index*self.batch_size + row]
            yawControlPoints = np.array(yawControlPoints)

            markersData_batch.append(markersDataNormalized)
            twistData_batch.append(twistData_normalized)
            positionControlPoints_batch.append(positionControlPoints)
            yawControlPoints_batch.append(yawControlPoints)

        markersData_batch = np.array(markersData_batch)
        twistData_batch = np.array(twistData_batch)
        positionControlPoints_batch = np.array(positionControlPoints_batch)
        yawControlPoints_batch = np.array(yawControlPoints_batch)

        if not self.sampleWeightEnabled:
            return ([markersData_batch, twistData_batch], [positionControlPoints_batch, yawControlPoints_batch])
        else:
            sampleWeight_batch = np.array(sampleWeight_batch)
            return ([markersData_batch, twistData_batch], [positionControlPoints_batch, yawControlPoints_batch], sampleWeight_batch)

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
        'numOfTwistSequence': 40,
        'maxTwistValues': [7, 4, 2.5, 0.4]
    }

    df = pd.read_pickle(directory)
    df = df.sample(frac=0.01)

    Xset = [df['markersData'].tolist(), df['vel'].tolist()]
    Yset = [df['positionControlPoints'].tolist(), df['yawControlPoints'].tolist()]
    imageList = df['images'].tolist()
    batchSize = 1
    inputImageShape=(480, 640, 3)
    markersDataReverseFactor = np.array([inputImageShape[1], inputImageShape[0], 1], dtype=np.float32)
    dataGen = MarkersAndTwistDataToBeizerDataGenerator(Xset, Yset, batchSize, inputImageShape, config, imageList)
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

def test_MarkersAndTwistDataToBezierDataGeneratorWithNomalization(directory):
    # configuration file:
    config = {
        'alpha': 0.1,
        'dataAugmentationRate': 0.0,
        'numOfImageSequence': 3,
        'markersNetworkType': 'LSTM',  # 'Dense'
        'twistNetworkType': 'LSTM',
        'twistDataGenType': 'Sequence',
        'numOfTwistSequence': 40,
        'maxTwistValues': [7, 4, 2.5, 0.4]
    }

    df = pd.read_pickle(directory)
    df = df.sample(frac=0.01)

    Xset = [df['markersData'].tolist(), df['vel'].tolist()]
    Yset = [df['positionControlPoints'].tolist(), df['yawControlPoints'].tolist()]
    batchSize = 1
    inputImageShape=(480, 640, 3)
    markersDataReverseFactor = np.array([inputImageShape[1], inputImageShape[0], 1], dtype=np.float32)
    dataGen = MarkersAndTwistDataToBeizerDataGenerator(Xset, Yset, batchSize, inputImageShape, config)
    imageList = dataGen.getImageList()

    for i in range(dataGen.__len__()):
        Xbatch, Ybatch = dataGen.__getitem__(i)
        print('working on batch #{}'.format(i))
        print(Xbatch)
        continue
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
    allDataFileWithMarkers = '/home/majd/catkin_ws/src/basic_rl_agent/data2/flightgoggles/datasets/imageBezierDataV2_1/allData_WITH_STATES_PROB_imageBezierDataV2_1_20220407-1358.pkl'
    test_MarkersAndTwistDataToBezierDataGeneratorWithNomalization(allDataFileWithMarkers)
   

if __name__ == '__main__':
    main()