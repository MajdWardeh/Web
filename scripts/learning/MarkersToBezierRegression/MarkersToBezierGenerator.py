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

class MarkersImagesToBeizerDataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size, targetImageShape, inputImageShape, segma=7, imageSet=None):
        '''
            @param x_set: a list that contains the markersData in an image. each markersData is an np array with shape=(4, 3)
            @param y_set: a list of two lists that contains the control points of the position and yaw respectively.
            @param targetImageShape: the target size of the images to be loaded. tuple that looks like (h, w, 3).
            @param inputImageShape: the size of the images that markersData are stored for. Useful to compute the markersData ratio.
        '''
        self.x_set = x_set
        self.positionControlPointsList, self.yawControlPointsList = y_set[0], y_set[1]
        self.batch_size = batch_size
        self.h, self.w  = targetImageShape[0], targetImageShape[1]
        self.markersPreprocessing = ImageMarkersGroundTruthPreprocessing(targetImageShape, inputImageShape, cornerSegma=segma)
        # remove the data with zeros markers
        self.__removeZerosMarkers()

        # debugging:
        assert not imageSet is None, 'imageSet is None'
        self.imageSet = imageSet

    def __removeZerosMarkers(self):
        remove_indices = []
        for idx, markersData in enumerate(self.x_set):
            if (markersData[:, -1] == 0).any(): # check if the Z component of any marker is zeros.
                remove_indices.append(idx)
        x_set = []
        positionCpList = []
        yawCpList = []
        for i in range(len(self.x_set)):
            if not i in remove_indices:
                x_set.append(self.x_set[i])
                positionCpList.append(self.positionControlPointsList[i])
                yawCpList.append(self.yawControlPointsList[i])
        self.x_set = x_set            
        self.positionControlPointsList = positionCpList
        self.yawControlPointsList = yawCpList
    
    def __len__(self):
        return math.ceil(len(self.x_set) / self.batch_size)

    def __getitem__(self, index):
        '''
            Generates data containing batch_size samples 
            @return a list: 1. corners_image_batch for x. 
                            2. [positionControlPoints_batch, yawControlPoints_batch] for y.
        '''
        cornersImages_batch = []
        positionControlPoints_batch = []
        
        ___images_batch = []

        for row in range(min(self.batch_size, len(self.x_set)-index*self.batch_size)):
            markersData = self.x_set[index*self.batch_size + row]
            assert (markersData[:, -1] != 0).any(), 'markersData have Z component euqals to zero' # check if the Z component of any marker is zeros.
            cornerFourImages = self.markersPreprocessing.computeGroundTruthCorners(markersData)
            #TODO process cornerImage: output: one image, thresholded
            cornerImage = np.sum(cornerFourImages, axis=-1)
            print('cornerFourImages.shape', cornerFourImages.shape, 'cornerImage.shape', cornerImage.shape)

            positionControlPoints = self.positionControlPointsList[index*self.batch_size + row]

            ___image = cv2.imread(self.imageSet[index*self.batch_size + row])
            ___image = cv2.resize(___image, (self.w, self.h))

            cornersImages_batch.append(cornerImage)
            positionControlPoints_batch.append(positionControlPoints)
            ___images_batch.append(___image)

        # cornersImages_batch = np.array(cornersImages_batch)
        # positionControlPoints_batch = np.array(positionControlPoints_batch)

        # Normalize inputs: conrnerImages are already normalized
        return (cornersImages_batch, positionControlPoints_batch, ___images_batch)


class MarkersDataToBeizerDataGenerator(Sequence):
    # TODO include yaw control points
    def __init__(self, x_set, y_set, batch_size, inputImageShape):
        '''
            @param x_set: a list that contains the markersData in an image. each markersData is an np array with shape=(4, 3)
            @param y_set: a list of two lists that contains the control points of the position and yaw respectively.
            @param inputImageShape: the size of the images that markersData are stored for. Useful to compute the markersData ratio.
        '''
        self.x_set = x_set
        self.positionControlPointsList, self.yawControlPointsList = y_set[0], y_set[1]
        self.batch_size = batch_size

        #compute markersDataFactor
        inputImageHeight, inputImageWidth, _ = inputImageShape
        self.markersDataFactor = np.array([1.0/float(inputImageWidth), 1.0/float(inputImageHeight), 1.0])

        # remove the data with zeros markers
        self.__removeZerosMarkers()

    def __removeZerosMarkers(self):
        remove_indices = []
        for idx, markersData in enumerate(self.x_set):
            if (markersData[:, -1] == 0).any(): # check if the Z component of any marker is zeros.
                remove_indices.append(idx)
        x_set = []
        positionCpList = []
        yawCpList = []
        for i in range(len(self.x_set)):
            if not i in remove_indices:
                x_set.append(self.x_set[i])
                positionCpList.append(self.positionControlPointsList[i])
                yawCpList.append(self.yawControlPointsList[i])
        self.x_set = x_set            
        self.positionControlPointsList = positionCpList
        self.yawControlPointsList = yawCpList

    def getIndex(self, index):
        '''
            for debugging only
        '''
        x = range(min(self.batch_size, len(self.x_set)-index*self.batch_size))
        return (x, self.batch_size)

    
    def __len__(self):
        return math.ceil(len(self.x_set) / self.batch_size)

    def __getitem__(self, index):
        '''
            Generates data containing batch_size samples 
            @return a list: 1. markersData_batch for x. 
                            2. positionControlPoints_batch for y.
                            TODO: include yaw control points
        '''
        markersData_batch = []
        positionControlPoints_batch = []
        yawControlPoints_batch = []

        for row in range(min(self.batch_size, len(self.x_set)-index*self.batch_size)):
            markersData = self.x_set[index*self.batch_size + row]
            assert (markersData[:, -1] != 0).any(), 'markersData have Z component euqals to zero' # check if the Z component of any marker is zeros.
            # normailze markersData:
            markersDataNormalized = np.multiply(markersData, self.markersDataFactor)
            markersDataNormalized = markersDataNormalized.reshape(12, )

            positionControlPoints = self.positionControlPointsList[index*self.batch_size + row]
            positionControlPoints = np.array(positionControlPoints).reshape(15, )

            yawControlPoints = self.yawControlPointsList[index*self.batch_size + row]
            yawControlPoints = np.array(yawControlPoints)

            markersData_batch.append(markersDataNormalized)
            positionControlPoints_batch.append(positionControlPoints)
            yawControlPoints_batch.append(yawControlPoints)


        markersData_batch = np.array(markersData_batch)
        positionControlPoints_batch = np.array(positionControlPoints_batch)
        yawControlPoints_batch = np.array(yawControlPoints_batch)
        return (markersData_batch, [positionControlPoints_batch, yawControlPoints_batch])

class MarkersAndTwistDataToBeizerDataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size, inputImageShape):
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

        # remove the data with zeros markers
        self.__removeZerosMarkers()

    def __removeZerosMarkers(self):
        remove_indices = []
        for idx, markersData in enumerate(self.markersDataSet):
            if (markersData[:, -1] == 0).any(): # check if the Z component of any marker is zeros.
                remove_indices.append(idx)
        markersDataTmpList = []
        twistDataTmpList = []
        positionCpList = []
        yawCpList = []
        for i in range(len(self.markersDataSet)):
            if not i in remove_indices:
                markersDataTmpList.append(self.markersDataSet[i])
                twistDataTmpList.append(self.twistDataSet[i])
                positionCpList.append(self.positionControlPointsList[i])
                yawCpList.append(self.yawControlPointsList[i])
        self.markersDataSet = markersDataTmpList            
        self.twistDataSet = twistDataTmpList
        self.positionControlPointsList = positionCpList
        self.yawControlPointsList = yawCpList

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
            Generates data containing batch_size samples 
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
            markersDataNormalized = markersDataNormalized.reshape(12, )

            twistData = self.twistDataSet[index*self.batch_size + row]

            twistData = twistData.reshape(400,) #np.concatenate([twistData[-1], twistData[-2]], axis=0)

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

        # twist data function selection
        twistDataGenType = self.config['twistDataGenType']
        self.twistDataGenFunction = None
        if twistDataGenType == 'last2points_and_EMA':
            self.twistDataGenFunction = self.__genTwistData_3point_last2_and_average
        elif twistDataGenType == 'EMA':
            self.twistDataGenFunction = self.__genTwistData_eponentialMovingAverage
        else:
            raise NotImplementedError

        self.imageList = imageList

        # remove the data with zeros markers
        self.__removeZerosMarkers()

        # remove the twistData that is none
        self.__removeNanTwistData()

    def __removeZerosMarkers(self):
        remove_indices = []
        for idx, markersData in enumerate(self.markersDataSet):
            if (markersData[:, -1] == 0).any(): # check if the Z component of any marker is zeros.
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

    def __removeNanTwistData(self):
        remove_indices = []
        for idx, twistData in enumerate(self.twistDataSet):
            if np.isnan(twistData).any():
                remove_indices.append(idx)
        markersDataTmpList = []
        twistDataTmpList = []
        positionCpList = []
        yawCpList = []
        imageList = [] if not self.imageList is None else None
        for i in range(len(self.twistDataSet)):
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
            if np.random.rand() <= self.dataAugmentingRate:
                randomIdx = np.random.randint(0, 4)
                markersDataNormalized[randomIdx] = np.array([0, 0, 0])

            markersDataNormalized = markersDataNormalized.reshape(12, )

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

    def __genTwistData_3point_last2_and_average(self, twistData):
        '''
            @returns 3 twist data points: the last 2 points and the exponential moving average (EMA) 
        '''
        alpha = self.config['alpha']
        averageTwist = twistData[0]
        for currTwist in twistData[1:]:
            averageTwist = alpha * currTwist + (1-alpha) * averageTwist
        return np.concatenate([twistData[-1], twistData[-2], averageTwist], axis=0) 

    def __genTwistData_eponentialMovingAverage(self, twistData):
        '''
            @returns the exponential moving average (EMA) of the twistData
        '''
        alpha = self.config['alpha']
        averageTwist = twistData[0]
        for currTwist in twistData[1:]:
            averageTwist = alpha * currTwist + (1-alpha) * averageTwist
        return averageTwist

def test_MarekrsImagesToBezierDataGenerator(directory):
    df = pd.read_pickle(directory)
    df = df.sample(frac=0.1)
    Xset = df['markersData'].tolist()
    Yset = [df['positionControlPoints'].tolist(), df['yawControlPoints'].tolist()]
    imageSet = [np.array2string(image_np[0, 0])[1:-1] for image_np in df['images'].tolist()]
    batchSize = 1
    dataGen = MarkersImagesToBeizerDataGenerator(Xset, Yset, batchSize, targetImageShape=(480, 640, 3), inputImageShape=(480, 640, 3), segma=7, imageSet=imageSet)
    print(dataGen.__len__())
    for i in range(dataGen.__len__()):
        Xbatch, Ybatch, imageBatch = dataGen.__getitem__(i)
        print('working on batch #{}'.format(i))
        for idx, im in enumerate(imageBatch):
            cornerImage = Xbatch[idx]
            cv2.imshow('input image', im)
            cv2.imshow('corners Image', cornerImage)
            cv2.waitKey(1000)

def test_MarkersDataToBezierDataGenerator(directory):
    df = pd.read_pickle(directory)
    df = df.sample(frac=0.1)
    Xset = df['markersData'].tolist()
    Yset = [df['positionControlPoints'].tolist(), df['yawControlPoints'].tolist()]
    imageSet = [np.array2string(image_np[0, 0])[1:-1] for image_np in df['images'].tolist()]
    batchSize = 1
    inputImageShape=(480, 640, 3)
    markersDataReverseFactor = np.array([inputImageShape[1], inputImageShape[0], 1], dtype=np.float32)
    dataGen = MarkersDataToBeizerDataGenerator(Xset, Yset, batchSize, inputImageShape)
    bezierVisulizer = BezierVisulizer(plot_delay=2)
    for i in range(dataGen.__len__()):
        Xbatch, Ybatch = dataGen.__getitem__(i)
        print('twistData.shape = ', Xbatch[1][0].shape)
        print('working on batch #{}'.format(i))
        x_range, bs = dataGen.getIndex(i)
        imageBatch = [imageSet[i*bs+k] for k in x_range]
        for idx, im in enumerate(imageBatch):
            markersDataNormalized = Xbatch[idx]
            markersDataNormalized = markersDataNormalized.reshape(4, 3)
            markersData = np.multiply(markersDataNormalized, markersDataReverseFactor)
            image = cv2.imread(im)
            for marker in markersData[:, :-1]:
                marker = marker.astype(np.int)
                image = cv2.circle(image, (marker[0], marker[1]), radius=6, color=(255, 0, 0), thickness=-1)
            positonCP, yawCP = Ybatch[0], Ybatch[1]
            positonCP = positonCP.reshape(5, 3).T
            bezierVisulizer.plotBezier(image, positonCP, yawCP)

def test_MarkersAndTwistDataToBezierDataGenerator(directory):
    df = pd.read_pickle(directory)
    df = df.sample(frac=0.1)
    Xset = [df['markersData'].tolist(), df['vel'].tolist()]
    Yset = [df['positionControlPoints'].tolist(), df['yawControlPoints'].tolist()]
    imageSet = [np.array2string(image_np[0, 0])[1:-1] for image_np in df['images'].tolist()]
    batchSize = 10
    inputImageShape=(480, 640, 3)
    markersDataReverseFactor = np.array([inputImageShape[1], inputImageShape[0], 1], dtype=np.float32)
    dataGen = MarkersAndTwistDataToBeizerDataGenerator(Xset, Yset, batchSize, inputImageShape)
    bezierVisulizer = BezierVisulizer(plot_delay=2)
    for i in range(dataGen.__len__()):
        Xbatch, Ybatch = dataGen.__getitem__(i)
        print('working on batch #{}'.format(i))
        x_range, bs = dataGen.getIndex(i)
        imageBatch = [imageSet[i*bs+k] for k in x_range]
        for idx, im in enumerate(imageBatch):
            print('xbatch.shape:', Xbatch[0].shape, Xbatch[1].shape)
            markersDataNormalized = Xbatch[0][idx]
            markersDataNormalized = markersDataNormalized.reshape(4, 3)
            markersData = np.multiply(markersDataNormalized, markersDataReverseFactor)
            twistData = Xbatch[1][idx]
            print(twistData)
            image = cv2.imread(im)
            for marker in markersData[:, :-1]:
                marker = marker.astype(np.int)
                image = cv2.circle(image, (marker[0], marker[1]), radius=6, color=(255, 0, 0), thickness=-1)
            positonCP, yawCP = Ybatch[0][idx], Ybatch[1][idx]
            print('Ybatch.shape:', positonCP.shape, yawCP.shape)
            positonCP = positonCP.reshape(5, 3).T
            yawCP = yawCP.reshape(1, 3)
            bezierVisulizer.plotBezier(image, positonCP, yawCP)
            
def test_MarkersAndTwistDataToBezierDataGeneratorWithDataAugmentation(directory):
    # configuration file:
    config = {
        'alpha': 0.1,
        'dataAugmentationRate': 0.9,
        'twistDataGenType': 'EMA'
    }

    df = pd.read_pickle(directory)
    df = df.sample(frac=0.5)
    Xset = [df['markersData'].tolist(), df['vel'].tolist()]
    Yset = [df['positionControlPoints'].tolist(), df['yawControlPoints'].tolist()]
    imageSet = [np.array2string(image_np[0, 0])[1:-1] for image_np in df['images'].tolist()]
    batchSize = 1
    inputImageShape=(480, 640, 3)
    markersDataReverseFactor = np.array([inputImageShape[1], inputImageShape[0], 1], dtype=np.float32)
    dataGen = MarkersAndTwistDataToBeizerDataGeneratorWithDataAugmentation(Xset, Yset, batchSize, inputImageShape, config, imageSet)
    imageSet = dataGen.getImageList()
    bezierVisulizer = BezierVisulizer(plot_delay=0.01)
    for i in range(dataGen.__len__()):
        Xbatch, Ybatch = dataGen.__getitem__(i)
        print('working on batch #{}'.format(i))
        x_range, bs = dataGen.getIndex(i)
        # imageBatch = [imageSet[i*bs+k] for k in x_range]
        imageBatch = [imageSet[i]]
        for idx, im in enumerate(imageBatch):
            # print('xbatch.shape:', Xbatch[0].shape, Xbatch[1].shape)

            # twist data:
            twistData = Xbatch[1][idx]
            print('twistData:')
            print(twistData.reshape(-1, 4))
            print()

            # markers data:
            markersDataNormalized = Xbatch[0][idx]
            markersDataNormalized = markersDataNormalized.reshape(4, 3)

            # testing markersData data augmentation:
            if (markersDataNormalized == 0).any():
                print('markersData has zeros', markersDataNormalized)

            markersData = np.multiply(markersDataNormalized, markersDataReverseFactor)
            image = cv2.imread(im)
            for marker in markersData[:, :-1]:
                marker = marker.astype(np.int)
                image = cv2.circle(image, (marker[0], marker[1]), radius=6, color=(255, 0, 0), thickness=-1)
            positonCP, yawCP = Ybatch[0][idx], Ybatch[1][idx]
            print('Ybatch.shape:', positonCP.shape, yawCP.shape)
            positonCP = positonCP.reshape(5, 3).T
            yawCP = yawCP.reshape(1, 3)
            bezierVisulizer.plotBezier(image, positonCP, yawCP)
            


def main():
    allDataFileWithMarkers = '/home/majd/catkin_ws/src/basic_rl_agent/data/debugging_data3/allDataWithMarkers.pkl'
    test_MarkersAndTwistDataToBezierDataGeneratorWithDataAugmentation(allDataFileWithMarkers)
   

if __name__ == '__main__':
    main()