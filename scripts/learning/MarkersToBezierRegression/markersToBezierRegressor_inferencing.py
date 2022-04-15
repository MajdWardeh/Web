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
from learning.MarkersToBezierRegression.markersToBezierRegressor_configurable_withCostumLoss import Network

class MarkersAndTwistDataToBeizerInferencer:
    def __init__(self, inputImageShape, config, networkWeightsFile):
        #compute markersDataFactor
        inputImageHeight, inputImageWidth, _ = inputImageShape
        self.markersDataFactor = np.array([1.0/float(inputImageWidth), 1.0/float(inputImageHeight), 1.0])
        self.config = config
        self.__processConfig()

        self.model = Network(self.config).getModel()
        if networkWeightsFile.endswith('ckpt'):
            self.model.load_weights(networkWeightsFile).expect_partial()
        else:
            self.model.load_weights(networkWeightsFile)

    ############################################ end of __init__ function

    def __processConfig(self):

        # normalization data (mean and std)
        self.markersData_hardcoded_mean = self.config.get('markersData_hardcoded_mean', \
                    np.array([322.00710606, 176.19885206, 12.77271492]))
        self.markersData_hardcoded_std = self.config.get('markersData_hardcoded_std', \
                    np.array([124.70433658, 73.10797561, 5.49590978]))
        self.twistData_hardcoded_mean = self.config.get('twistData_hardcoded_mean', \
                    np.array([3.5, 0., 1.25, 0.]))
        self.twistData_hardcoded_std = self.config.get('twistData_hardcoded_std', \
                    np.array([2.39911219, 1.36576634, 0.68722698, 0.10353576]))

        # markers data shape
        self.numOfImageSequence = self.config['numOfImageSequence']
        self.markersNetworkType = self.config['markersNetworkType'] # Dense, LSTM
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

    def inference(self, markersDataRaw, twistDataRaw):
        markersData_reshaped = markersDataRaw.reshape(-1, 3)
        markersData_reshaped_normalized = (markersData_reshaped - self.markersData_hardcoded_mean) / self.markersData_hardcoded_std
        markersData_normalized_reshaped_again = markersData_reshaped_normalized.reshape(self.markersDataShape)
        markersDataInput = markersData_normalized_reshaped_again[np.newaxis, :]

        twistData_normalized = (twistDataRaw - self.twistData_hardcoded_mean) / self.twistData_hardcoded_std
        twistData = self.twistDataGenFunction(twistData_normalized)
        twistDataInput = twistData[np.newaxis, :]

        networkInput = [markersDataInput, twistDataInput]
        # network inferencing:
        y_hat = self.model(networkInput, training=False)

        # important when the model has LSTM architecture
        self.model.reset_states()

        return y_hat

    def old_normalizing_inference(self, markersDataRaw, twistDataRaw):
        markersDataNormalized = np.multiply(markersDataRaw, self.markersDataFactor)
        markersDataNormalizedReshaped = markersDataNormalized.reshape(self.markersDataShape)
        markersDataInput = markersDataNormalizedReshaped[np.newaxis, :]

        twistData = self.twistDataGenFunction(twistDataRaw)
        twistDataInput = twistData[np.newaxis, :]

        networkInput = [markersDataInput, twistDataInput]
        # network inferencing:
        y_hat = self.model(networkInput, training=False)

        return y_hat
    
    def reset_states(self):
        self.model.reset_states()

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

def main():
    pass

if __name__ == '__main__':
    main()