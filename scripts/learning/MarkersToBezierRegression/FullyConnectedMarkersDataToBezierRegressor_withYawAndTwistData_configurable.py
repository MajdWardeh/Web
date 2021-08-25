import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import math
import cv2
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Input, layers, Model, backend as k
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow.keras.metrics as metrics
from tensorflow.keras.utils import Sequence
from keras.callbacks import TensorBoard
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.losses import Loss, MeanAbsoluteError, MeanSquaredError
from MarkersToBezierGenerator import MarkersAndTwistDataToBeizerDataGenerator, MarkersAndTwistDataToBeizerDataGeneratorWithDataAugmentation

from Bezier_untils import BezierVisulizer, bezier4thOrder

class Network:

    def __init__(self, config):
        self.numOfDenseLayers = config['numOfDenseLayers']  # int 
        self.numOfUnitsPerLayer = config['numOfUnitsPerLayer'] # list default 150, 80, 80
        self.dropRatePerLayer = config['dropRatePerLayer'] # list default 0.3, 0.3, 0

        # twist data
        self.twistDataInputShape = None
        twistDataGenType = config['twistDataGenType']
        if twistDataGenType == 'last2points_and_EMA':
            self.twistDataInputShape = (3*4, )
        elif twistDataGenType == 'EMA':
            self.twistDataInputShape = (4, )
        else:
            raise NotImplementedError

        self.model = self._createModel()

    def getModel(self):
        return self.model

    def _createModel(self):
        markersDataInput = layers.Input(shape=(12, ))
        twistDataInput = layers.Input(shape=self.twistDataInputShape)
        x = layers.concatenate([markersDataInput, twistDataInput], axis=1)

        for i in range(self.numOfDenseLayers):
            x = layers.Dense(self.numOfUnitsPerLayer[i], activation='relu')(x)
            if self.dropRatePerLayer[i] != 0:
                x = layers.Dropout(self.dropRatePerLayer[i])(x)

        positionOutput = layers.Dense(15, activation='linear', name='positionOutput')(x)
        yawOutput = layers.Dense(3, activation='linear', name='yawOutput')(x)

        model = Model(inputs=[markersDataInput, twistDataInput], outputs=[positionOutput, yawOutput])
        return model

class Training:

    def __init__(self, config):
        print(config)
        self.config = config
        self.learningRate = config['learningRate'] # default 0.0005
        self.configNum = config['configNum'] # int
        self.numOfEpochs = config['numOfEpochs'] # int
        if 'epochLearningRateRules' in config:
            self.epochLearningRateRules = config['epochLearningRateRules'] # list of tuples
        else:
            self.epochLearningRateRules = None

        self.model = Network(config).getModel()
        self.model.summary()
        self.trainBatchSize, self.testBatchSize = 1000, 1000 #500, 500
        self.trainGen, self.testGen = self.createTrainAndTestGeneratros(self.trainBatchSize, self.testBatchSize)
        self.model_weights_dir = '/home/majd/catkin_ws/src/basic_rl_agent/data/deep_learning/MarkersToBezierDataFolder/models_weights'
        self.saveHistoryDir = '/home/majd/catkin_ws/src/basic_rl_agent/data/deep_learning/MarkersToBezierDataFolder/trainHistoryDict'

        self.model.compile(
            optimizer=Adam(learning_rate=self.learningRate),
            loss={'positionOutput': 'mean_squared_error', 'yawOutput': 'mean_squared_error'},
            # metrics=[metrics.MeanSquaredError(name='mse'), metrics.MeanAbsoluteError(name='mae')])
            metrics={'positionOutput': metrics.MeanAbsoluteError(), 'yawOutput':metrics.MeanAbsoluteError()})
    
    def createTrainAndTestGeneratros(self, trainBatchSize, testBatchSize):
        allDataFileWithMarkers = '/home/majd/catkin_ws/src/basic_rl_agent/data/debugging_data3/allDataWithMarkers.pkl'
        inputImageShape = (480, 640, 3) 
        df = pd.read_pickle(allDataFileWithMarkers) 
        # randomize the data
        df = df.sample(frac=1, random_state=1)
        df.reset_index(drop=True, inplace=True)

        self.train_df = df.sample(frac=0.8, random_state=10)
        self.test_df = df.drop(labels=self.train_df.index, axis=0)
        train_Xset, train_Yset = [self.train_df['markersData'].tolist(), self.train_df['vel'].tolist()], [self.train_df['positionControlPoints'].tolist(), self.train_df['yawControlPoints'].tolist()]
        test_Xset, test_Yset = [self.test_df['markersData'].tolist(), self.test_df['vel'].tolist()], [self.test_df['positionControlPoints'].tolist(), self.test_df['yawControlPoints'].tolist()]
        trainGenerator = MarkersAndTwistDataToBeizerDataGeneratorWithDataAugmentation(train_Xset, train_Yset, trainBatchSize, inputImageShape, self.config)
        testGenerator = MarkersAndTwistDataToBeizerDataGeneratorWithDataAugmentation(test_Xset, test_Yset, testBatchSize, inputImageShape, self.config)
        return trainGenerator, testGenerator

    def learningRateScheduler(self, epoch, lr):
        if epoch < self.epochLearningRateRules[0][0]:
            lr = self.epochLearningRateRules[0][1]
        elif  epoch < self.epochLearningRateRules[1][0]:
            lr = self.epochLearningRateRules[1][1]
        elif  epoch < self.epochLearningRateRules[2][0]:
            lr = self.epochLearningRateRules[2][1]
        return lr

    def trainModel(self):
        #TODO if training was less than 5 minutes, don't save weights.
        callbacks = []
        if not self.epochLearningRateRules is None:
            callbacks.append(tf.keras.callbacks.LearningRateScheduler(self.learningRateScheduler))
        try:
            history = self.model.fit(
                x=self.trainGen, epochs=self.numOfEpochs, 
                validation_data=self.testGen, validation_steps=5, 
                callbacks=callbacks,
                verbose=1, workers=4, use_multiprocessing=True)
            with open(os.path.join(self.saveHistoryDir, 'history_MarkersToBeizer_FC_scratch_withYawAndTwistData_config{}_{}.pkl'.format(self.configNum, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))), 'wb') as file_pi:
                pickle.dump(history.history, file_pi) 
        except KeyboardInterrupt:
            print('KeyboardInterrupt, model weights were saved.')
        finally:
            self.model.save_weights(os.path.join(self.model_weights_dir, 'weights_MarkersToBeizer_FC_scratch_withYawAndTwistData_config{}_{}.h5'.format(self.configNum, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))))

    def testModel(self):
        self.model.load_weights(os.path.join(self.model_weights_dir, 'weights_MarkersToBeizer_FC_scratch_withYawAndTwistData_config8_20210825-050351.h5'))
        _, testGen = self.createTrainAndTestGeneratros(1, 1)

        imageSet = [np.array2string(image_np[0, 0])[1:-1] for image_np in self.test_df['images'].tolist()]

        bezierVisulizer = BezierVisulizer(plot_delay=2)
        for k in range(1000, testGen.__len__())[:]:
            print(k)
            x, y = testGen.__getitem__(k)
            y_hat = self.model(x, training=False)
            positionCP, yawCP = y[0][0], y[1][0]
            positionCP_hat, yawCP_hat = y_hat[0][0].numpy(), y_hat[1][0].numpy()
            print(np.mean(np.abs(positionCP-positionCP_hat)), np.mean(np.abs(yawCP-yawCP_hat)))
            imageName = imageSet[k]
            image = cv2.imread(imageName)
            positionCP = positionCP.reshape(5, 3).T
            yawCP = yawCP.reshape(1, 3)
            positionCP_hat = positionCP_hat.reshape(5, 3).T
            yawCP_hat = yawCP_hat.reshape(1, 3)
            bezierVisulizer.plotBezier(image, positionCP, yawCP, positionCP_hat, yawCP_hat)


def train():
    configs = defineConfigs1()
    # training = Training(configs[6])
    # training.trainModel()

    for i in range(len(configs)):
        training = Training(configs[i])
        training.trainModel()

def test():
    configs = defineConfigs()
    training = Training(configs[6])
    training.testModel()

def defineConfigs():
    configTest = {
        'numOfDenseLayers': 3,
        'numOfUnitsPerLayer': [80, 40, 20],
        'dropRatePerLayer': [0.4, 0.2, 0], 
        'learningRate': 0.0005,
        'configNum': 'test',
        'numOfEpochs': 1
    }
    config0 = {
        'numOfDenseLayers': 3,
        'numOfUnitsPerLayer': [80, 50, 50],
        'dropRatePerLayer': [0.4, 0.2, 0], 
        'learningRate': 0.0005,
        'configNum': 0,
        'numOfEpochs': 1200
    }

    config1 = {
        'numOfDenseLayers': 3,
        'numOfUnitsPerLayer': [100, 80, 50],
        'dropRatePerLayer': [0.3, 0.2, 0], 
        'learningRate': 0.0005,
        'configNum': 1,
        'numOfEpochs': 1200
    }

    config2 = {
        'numOfDenseLayers': 2,
        'numOfUnitsPerLayer': [100, 80],
        'dropRatePerLayer': [0.3, 0.1], 
        'learningRate': 0.0005,
        'configNum': 2,
        'numOfEpochs': 1000
    }

    config3 = {
        'numOfDenseLayers': 3,
        'numOfUnitsPerLayer': [200, 100, 80],
        'dropRatePerLayer': [0.3, 0.3, 0.3], 
        'learningRate': 0.0005,
        'configNum': 3,
        'numOfEpochs': 2000
    }
    
    config4 = {
        'numOfDenseLayers': 4,
        'numOfUnitsPerLayer': [100, 80, 60, 40],
        'dropRatePerLayer': [0.4, 0.3, 0.2, 0.1], 
        'learningRate': 0.0005,
        'configNum': 4,
        'numOfEpochs': 2000
    }

    config5 = {
        'numOfDenseLayers': 3,
        'numOfUnitsPerLayer': [100, 80, 50],
        'dropRatePerLayer': [0.3, 0.3, 0.3], 
        'learningRate': 0.0005,
        'configNum': 5,
        'numOfEpochs': 1200
    }

    config6 = {
        'numOfDenseLayers': 3,
        'numOfUnitsPerLayer': [100, 80, 50],
        'dropRatePerLayer': [0, 0, 0], 
        'learningRate': 0.0005,
        'configNum': 6,
        'numOfEpochs': 1400
    }

    config7 = {
        'numOfDenseLayers': 3,
        'numOfUnitsPerLayer': [100, 80, 50],
        'dropRatePerLayer': [0, 0, 0], 
        'learningRate': 0.001,
        'configNum': 7,
        'numOfEpochs': 1400,
        'epochLearningRateRules': [(50, 0.001), (100, 0.0005), (300, 0.0002), (500, 0.0001)]
    }

    configs = [config0, config1, config2, config3, config4, config5, config6, config7]
    # configs = [configTest]
    return configs

def defineConfigs1():
    '''
        for training with different exponential moving average parameter alpha
    '''
    config8 = {
        'numOfDenseLayers': 3,
        'numOfUnitsPerLayer': [100, 80, 50],
        'dropRatePerLayer': [0, 0, 0], 
        'learningRate': 0.0005,
        'configNum': 8,
        'numOfEpochs': 1600,
        'alpha': 0.1,
        'dataAugmentationRate': 0.0
    }

    config9 = {
        'numOfDenseLayers': 3,
        'numOfUnitsPerLayer': [100, 80, 50],
        'dropRatePerLayer': [0, 0, 0], 
        'learningRate': 0.0005,
        'configNum': 9,
        'numOfEpochs': 1600,
        'alpha': 0.2,
        'dataAugmentationRate': 0.0
    }

    config10 = {
        'numOfDenseLayers': 3,
        'numOfUnitsPerLayer': [100, 80, 50],
        'dropRatePerLayer': [0, 0, 0], 
        'learningRate': 0.0005,
        'configNum': 10,
        'numOfEpochs': 1600,
        'alpha': 0.3,
        'dataAugmentationRate': 0.0
    }

    config11 = {
        'numOfDenseLayers': 3,
        'numOfUnitsPerLayer': [100, 80, 50],
        'dropRatePerLayer': [0, 0, 0], 
        'learningRate': 0.0005,
        'configNum': 11,
        'numOfEpochs': 1600,
        'alpha': 0.4,
        'dataAugmentationRate': 0.0
    }

    config12 = {
        'numOfDenseLayers': 3,
        'numOfUnitsPerLayer': [100, 80, 50],
        'dropRatePerLayer': [0, 0, 0], 
        'learningRate': 0.0005,
        'configNum': 12,
        'numOfEpochs': 1600,
        'alpha': 0.5,
        'dataAugmentationRate': 0.0
    }

    config13 = {
        'numOfDenseLayers': 3,
        'numOfUnitsPerLayer': [90, 70, 40],
        'dropRatePerLayer': [0, 0, 0], 
        'learningRate': 0.0005,
        'configNum': 13,
        'numOfEpochs': 1300,
        'alpha': 0.5,
        'dataAugmentationRate': 0.0,
        'twistDataGenType': 'EMA'
    }
    config14 = {
        'numOfDenseLayers': 3,
        'numOfUnitsPerLayer': [100, 80, 50],
        'dropRatePerLayer': [0, 0, 0], 
        'learningRate': 0.0005,
        'configNum': 14,
        'numOfEpochs': 1300,
        'alpha': 0.5,
        'dataAugmentationRate': 0.0,
        'twistDataGenType': 'EMA'
    }
    configs = [config13, config14]
    return configs


if __name__ == '__main__':
    train()
    # test()