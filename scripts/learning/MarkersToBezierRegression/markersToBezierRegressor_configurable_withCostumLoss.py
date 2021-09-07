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
import yaml
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
from tensorflow.keras import Input, layers, Model, backend as k
from tensorflow.keras.layers import Conv1D, LeakyReLU, Flatten, Dense
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow.keras.metrics as metrics
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.losses import Loss, MeanAbsoluteError, MeanSquaredError
from MarkersToBezierGenerator import  MarkersAndTwistDataToBeizerDataGeneratorWithDataAugmentation
from untils.configs_utils import loadAllConfigs

from Bezier_untils import BezierVisulizer, bezier4thOrder
from BezierLossFunction import BezierLoss


class Network:

    def __init__(self, config):
        self.config = config
        self.numOfDenseLayers = config['numOfDenseLayers']  # int 
        self.numOfUnitsPerLayer = config['numOfUnitsPerLayer'] # list default 150, 80, 80
        self.dropRatePerLayer = config['dropRatePerLayer'] # list default 0.3, 0.3, 0

        # image sequence configs
        self.numOfImageSequence = config['numOfImageSequence']
        self.markersNetworkType = config['markersNetworkType']
        if self.markersNetworkType == 'LSTM':
            self.markers_LSTM_units = config['markers_LSTM_units']

        # twist data configs
        self.twistNetworkType = config['twistNetworkType']
        if self.twistNetworkType == 'LSTM':
            self.twist_LSTM_units = config['twist_LSTM_units']

        twistDataGenType = config['twistDataGenType']
        if twistDataGenType == 'last2points_and_EMA':
            self.twistDataInputShape = (3*4, )
        elif twistDataGenType == 'last2points':
            self.twistDataInputShape = (2*4, )
        elif twistDataGenType == 'EMA':
            self.twistDataInputShape = (4, )
        elif twistDataGenType == 'Sequence':
            if self.twistNetworkType == 'LSTM':
                self.numOfTwistSequence = config['numOfTwistSequence']
                self.twistDataInputShape = (self.numOfTwistSequence, 4)
            elif self.twistNetworkType == 'Conv':
                self.numOfTwistSequence = config['numOfTwistSequence']
                self.twistDataInputShape = (self.numOfTwistSequence, 4)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        self.model = self._createModel()

    def getModel(self):
        return self.model

    def _createModel(self):

        ### markers network type configurations
        if self.markersNetworkType == 'Dense':
            markersDataInput = layers.Input(shape=(self.numOfImageSequence * 12, ), name='markersDataInput')
            markersDataOut = markersDataInput
        elif self.markersNetworkType == 'Separate_Dense':
            markersSeparateDenseLayers = self.config['markersSeparateDenseLayers']
            markersDataInput = layers.Input(shape=(self.numOfImageSequence * 12, ), name='markersDataInput')
            x = markersDataInput
            for i in range(len(markersSeparateDenseLayers)):
                x = layers.Dense(markersSeparateDenseLayers[i], activation='relu')(x)
            markersDataOut = x
        elif self.markersNetworkType == 'LSTM':
            markersDataInput = layers.Input(shape=(self.numOfImageSequence, 12), name='markersDataInput')
            markersDataOut = layers.LSTM(self.markers_LSTM_units)(markersDataInput)
        else:
            raise NotImplementedError

        ### twist network type configurations
        twistDataInput = layers.Input(shape=self.twistDataInputShape, name='twistDataInput')

        if self.twistNetworkType == 'Dense':
            twistDataOut = twistDataInput
        elif self.twistNetworkType == 'Separate_Dense':
            twistSeparateDenseLayers = self.config['twistSeparateDenseLayers']
            x = twistDataInput
            for i in range(len(twistSeparateDenseLayers)):
                x = layers.Dense(twistSeparateDenseLayers[i], activation='relu')(x)
            twistDataOut = x

        elif self.twistNetworkType == 'LSTM':
            twistDataOut = layers.LSTM(self.twist_LSTM_units)(twistDataInput)

        elif self.twistNetworkType == 'Conv':
            g = self.config.get('convTwistNetworkG', 2.0)
            twist_conv_net = [Conv1D(int(64 * g), kernel_size=2, strides=1, padding='same',
                        dilation_rate=1),
                        LeakyReLU(alpha=1e-2),
                        Conv1D(int(32 * g), kernel_size=2, strides=1, padding='same', dilation_rate=1),
                        LeakyReLU(alpha=1e-2),
                        Conv1D(int(32 * g), kernel_size=2, strides=1, padding='same', dilation_rate=1),
                        LeakyReLU(alpha=1e-2),
                        Conv1D(int(32 * g), kernel_size=2, strides=1, padding='same', dilation_rate=1),
                        Flatten(),
                        Dense(int(10*g))]
            x = twistDataInput
            for f in twist_conv_net:
                x = f(x)
            twistDataOut = x

        else:
            raise NotImplementedError

        ### the reset of the network
        x = layers.concatenate([markersDataOut, twistDataOut], axis=1)

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

        self.lossFunction = self.config.get('lossFunction', 'mse')
        if self.lossFunction == 'mse':
            self.loss_dict = {'positionOutput': 'mean_squared_error', 'yawOutput': 'mean_squared_error'}
            self.metric_dict = {'positionOutput': metrics.MeanAbsoluteError(), 'yawOutput':metrics.MeanAbsoluteError()}
        elif self.lossFunction == 'BezierLoss':
            positionBezierLoss = BezierLoss(numOfControlPoints=5, dimentions=3)
            yawBezierLoss = BezierLoss(numOfControlPoints=3, dimentions=1)
            self.loss_dict = {'positionOutput': positionBezierLoss, 'yawOutput': yawBezierLoss}
            self.metric_dict = {'positionOutput': [metrics.MeanAbsoluteError(), metrics.MeanSquaredError()], 'yawOutput':[metrics.MeanAbsoluteError(), metrics.MeanSquaredError()]}
        else:
            raise NotImplementedError 

        self.model = Network(config).getModel()
        self.model.summary()

        ### Directory check and checkPointsDir create
        datasetPath = '/home/majd/catkin_ws/src/basic_rl_agent/data/allData_imageBezierData1_midPointData_20210908-0018.pkl'     #/allDataWithMarkers.pkl'
        print('dataset path:', datasetPath)
        self.datasetName = datasetPath.split('/')[-1].split('.pkl')[0]
        self.model_weights_dir = '/home/majd/catkin_ws/src/basic_rl_agent/data/deep_learning/MarkersToBezierDataFolder/models_weights'
        self.saveHistoryDir = '/home/majd/catkin_ws/src/basic_rl_agent/data/deep_learning/MarkersToBezierDataFolder/trainHistoryDict'
        for directory in [self.model_weights_dir, self.saveHistoryDir]:
            assert os.path.exists(directory), 'directory: {} was not found'.format(directory)

        self.model_final_name = 'config{}_{}_{:04d}_{}'.format(self.configNum, self.datasetName, self.numOfEpochs, datetime.datetime.now().strftime("%Y%m%d-%H%M"))
        if self.numOfEpochs >= 5:
            self.checkPointsDir = os.path.join(self.model_weights_dir, 'weights_{}'.format(self.model_final_name))
            os.mkdir(self.checkPointsDir)
        else:
            self.checkPointsDir = self.model_weights_dir

        self.trainBatchSize, self.testBatchSize = 1000, 1000 #500, 500
        self.trainGen, self.testGen = self.createTrainAndTestGeneratros(datasetPath, self.trainBatchSize, self.testBatchSize)

        self.model.compile(
            optimizer=Adam(learning_rate=self.learningRate),
            loss=self.loss_dict,
            # metrics=[metrics.MeanSquaredError(name='mse'), metrics.MeanAbsoluteError(name='mae')])
            metrics=self.metric_dict)
    
    def createTrainAndTestGeneratros(self, datasetPath, trainBatchSize, testBatchSize):
        inputImageShape = (480, 640, 3) 
        df = pd.read_pickle(datasetPath) 
        # randomize the data
        df = df.sample(frac=1, random_state=1)
        df.reset_index(drop=True, inplace=True)

        self.train_df = df.sample(frac=0.95, random_state=10)
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

    def trainModel(self, startFromCheckPointDict=None):
        if not startFromCheckPointDict is None:
            checkpointPath = startFromCheckPointDict.get('config{}'.format(self.configNum), None)
            print(checkpointPath)
            if not checkpointPath is None:
                self.model.load_weights(checkpointPath)
                checkpointName = checkpointPath.split('/')[-1].split('.pkl')[0]
                print('training config{}, starting from ckpt: {}'.format(self.configNum, checkpointName))
                final_name_splited = self.model_final_name.split('config{}'.format(self.configNum))
                ckpName = checkpointName.split('.h5')
                print(ckpName)
                self.model_final_name = 'config{}_ckpt_{}'.format(self.configNum, ''.join(ckpName))  + ''.join(final_name_splited[1:]) + '.h5'

        modelWeightsPath = os.path.join(self.model_weights_dir, 'wegihts_{}.h5'.format(self.model_final_name))
        modelHistoryPath = os.path.join(self.saveHistoryDir, 'history_{}.pkl'.format(self.model_final_name))


        #### callbacks definition
        callbacks = []
        if not self.epochLearningRateRules is None:
            callbacks.append(tf.keras.callbacks.LearningRateScheduler(self.learningRateScheduler))
        
        ## checkpoints callback
        saveEveryEpochsNum = self.config.get('saveEveryEpochsNum', 200)
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(self.checkPointsDir, 'weights_config{}_{}_{}.h5'.format(self.configNum, self.datasetName, '{epoch:04d}')),
            save_weights_only=True, save_freq= self.trainGen.__len__()*saveEveryEpochsNum)
        callbacks.append(model_checkpoint_callback)

        try:
            history = self.model.fit(
                x=self.trainGen, epochs=self.numOfEpochs, 
                validation_data=self.testGen, validation_steps=5, 
                callbacks=callbacks,
                verbose=1, workers=4, use_multiprocessing=True)
            with open(modelHistoryPath, 'wb') as file_pi:
                pickle.dump(history.history, file_pi) 
        except KeyboardInterrupt:
            print('KeyboardInterrupt, model weights were saved.')
        except Exception as e:
            print(e)
        finally:
            self.model.save_weights(modelWeightsPath)
            print('config{} is done.'.format(self.config['configNum']))

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


def train(configs, startFromCheckpointDict=None):
    # training = Training(configs[6])
    # training.trainModel()

    ## set the training to 1200
    # print('changing cofings in a bad way!')
    # for key in configs.keys():
        # config = configs[key]
        # config['lossFunction'] = 'BezierLoss'
        # config['configNum'] = '{}_BeizerLoss'.format(config['configNum'])
        # config['numOfEpochs'] = 20
        # configs[key] = config 

    for key in configs.keys():
        training = Training(configs[key])
        training.trainModel(startFromCheckpointDict)

def test(configs):
    training = Training(configs[6])
    training.testModel()

def defineConfigs():
    pass

def configsToFile():
    configs = defineConfigs()
    configsDict = {}
    for config in configs:
        configsDict['config{}'.format(config['configNum'])] = config
    with open('./configs/configs_{}.yaml'.format(int(time.time())), 'w') as outfile:
        yaml.dump(configsDict, outfile, default_flow_style=False) 

def loadConfigsFromFile(yaml_file):
    with open(yaml_file, 'r') as stream:
        try:
            loadedConfigs = yaml.load(stream, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)
    return loadedConfigs

def trainOnConfigs(configsRootDir):
    # listOfConfigNums = ['config15', 'config16', 'config17', 'config20']
    listOfConfigNums = []
    listOfConfigNums_colab0 = ['config17']
    listOfConfigNums_colab1 = ['config15']

    arg0 = sys.argv[1] if len(sys.argv) > 1 else ''
    if arg0 == '':
        pass
    elif arg0 == 'colab0':
        listOfConfigNums = listOfConfigNums_colab0
        print('colab0 is selected')
    elif arg0 == 'colab1':
        listOfConfigNums = listOfConfigNums_colab1
        print('colab1 is selected')
    elif arg0 == 'configs5':
        listOfConfigNums = ['config40', 'config41']
        print('working with configs5')
    elif arg0 == 'existed':
        pass
    else:
        listOfConfigNums = [arg0]
        
    print('listOfConfigNums:', listOfConfigNums)

    startFromCheckpoint = None # {
    #     'config17': '/home/majd/catkin_ws/src/basic_rl_agent/data/deep_learning/MarkersToBezierDataFolder/models_weights/wegihts_config17_BeizerLoss_imageToBezierData1_1800_20210905-1315.h5',
    #     'config61': '/home/majd/catkin_ws/src/basic_rl_agent/data/deep_learning/MarkersToBezierDataFolder/models_weights/wegihts_config61_imageToBezierData1_1800_20210906-1247.h5' 
    # }

    allConfigs = loadAllConfigs(configsRootDir, listOfConfigNums)
    train(allConfigs, startFromCheckpoint)

if __name__ == '__main__':
    trainOnConfigs(configsRootDir='/home/majd/catkin_ws/src/basic_rl_agent/scripts/learning/MarkersToBezierRegression/configs')

    # configs_file = '/home/majd/catkin_ws/src/basic_rl_agent/scripts/learning/MarkersToBezierRegression/configs/configs3.yaml'
    # configs = loadConfigsFromFile(configs_file)
    # train(configs)

    # # test()