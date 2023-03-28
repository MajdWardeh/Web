import sys
import gc
from turtle import position
import warnings
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
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
from tensorflow.keras import Input, layers, Model, backend as k
from tensorflow.keras.layers import Conv1D, LeakyReLU, Flatten, Dense
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow.keras.metrics as metrics
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.losses import Loss, MeanAbsoluteError, MeanSquaredError

from .MarkersToBezierGenerator import  MarkersAndTwistDataToBeizerDataGenerator
from .untils.configs_utils import loadAllConfigs

from Bezier_untils import BezierVisulizer, bezier4thOrder
from .BezierLossFunction import BezierLoss
from .customMetric import ControlPointsMetric


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
        def _createMarkersNetworkPath():
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
            return markersDataInput, markersDataOut
        
        def _createTwistNetworkPath():

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
            
            return twistDataInput, twistDataOut

        ###### the reset of the network ######
        markersDataInput, markersDataOut = _createMarkersNetworkPath()
        twistDataInput, twistDataOut = _createTwistNetworkPath()

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

    def __init__(self, config, evaluateOnly=False):
        print(config)
        self.config = config
        self.apply_sampleWeight = False
        warnings.warn('sampleWeight is applied: {}'.format(self.apply_sampleWeight))
        self.learningRate = config['learningRate'] # default 0.0005
        self.configNum = config['configNum'] # int
        self.numOfEpochs = config['numOfEpochs'] # int
        if 'epochLearningRateRules' in config:
            self.epochLearningRateRules = config['epochLearningRateRules'] # list of tuples
        else:
            self.epochLearningRateRules = None

        self.lossFunction = self.config.get('lossFunction', 'mse')
        positionCpMetric = ControlPointsMetric('positionCP')
        headingCpMetric = ControlPointsMetric('headingCP', numOfCp=3, dimention=1)
        if self.lossFunction == 'mse':
            self.loss_dict = {'positionOutput': 'mean_squared_error', 'yawOutput': 'mean_squared_error'}
            # self.metric_dict = {'positionOutput': [metrics.MeanAbsoluteError(), positionCpMetric], 'yawOutput':[metrics.MeanAbsoluteError(), headingCpMetric]}
        elif self.lossFunction == 'BezierLoss':
            positionBezierLoss = BezierLoss(numOfControlPoints=5, dimentions=3, config=config)
            yawBezierLoss = BezierLoss(numOfControlPoints=3, dimentions=1, config=config)
            self.loss_dict = {'positionOutput': positionBezierLoss, 'yawOutput': yawBezierLoss}
            # self.metric_dict = {'positionOutput': [metrics.MeanAbsoluteError(), metrics.MeanSquaredError(), positionCpMetric], 'yawOutput':[metrics.MeanAbsoluteError(), metrics.MeanSquaredError(), headingCpMetric]}
        else:
            raise NotImplementedError 
        
        self.metric_dict = {'positionOutput': [positionCpMetric], 'yawOutput':[headingCpMetric]}

        self.model = Network(config).getModel()
        self.model.summary()
        print('using {} as loss function'.format(self.lossFunction))

        ### Directory check and checkPointsDir create
        # datasetPath = '/home/majd/catkin_ws/src/basic_rl_agent/data2/flightgoggles/datasets/imageBezierData2/allData_imageBezierData2_20210909-1936.pkl'     #/allDataWithMarkers.pkl'     #/allDataWithMarkers.pkl'
        # datasetPath = '/home/majd/catkin_ws/src/basic_rl_agent/data2/flightgoggles/datasets/imageBezierData_I8_1000/allData_WITH_STATES_PROB_imageBezierData_I8_1000_20220418-1855.pkl'
        # datasetPath = '/home/majd/catkin_ws/src/basic_rl_agent/data2/flightgoggles/datasets/allData_imageBezierData1_midPointData_20210908-0018.pkl'
        # datasetPath = '/home/majd/catkin_ws/src/basic_rl_agent/data2/flightgoggles/datasets/imageBezier_updated_datasets/imageBezierData_1000_20FPS/allData_imageBezierData_1000_20FPS_20221225-0126.pkl'
        # datasetPath = '/home/majd/catkin_ws/src/basic_rl_agent/data2/flightgoggles/datasets/imageBezier_updated_datasets/imageBezierData_1000_30FPS/allData_imageBezierData_1000_30FPS_20221222-2337.pkl'
        datasetPath = '/home/majd/catkin_ws/src/basic_rl_agent/data2/flightgoggles/datasets/imageBezier_updated_datasets/imageBezierData_1000_DImg3_DTwst10/allData_imageBezierData_1000_DImg3_DTwst10_20230115-1336.pkl'

        print('dataset path:', datasetPath)
        self.datasetName = datasetPath.split('/')[-1].split('.pkl')[0]
        self.model_weights_dir = '/home/majd/catkin_ws/src/basic_rl_agent/data2/flightgoggles/deep_learning/MarkersToBezierDataFolder/models_weights'
        self.saveHistoryDir = '/home/majd/catkin_ws/src/basic_rl_agent/data2/flightgoggles/deep_learning/MarkersToBezierDataFolder/trainHistoryDict'
        for directory in [self.model_weights_dir, self.saveHistoryDir]:
            assert os.path.exists(directory), 'directory: {} was not found'.format(directory)
        sampleWeight_name = 'sampleWeightApplied' if self.apply_sampleWeight else 'NoSampleWeight'
        if not evaluateOnly:
            self.model_final_name = 'config{}_{}_{}_{:04d}_{}'.format(self.configNum, self.datasetName, sampleWeight_name, self.numOfEpochs, datetime.datetime.now().strftime("%Y%m%d-%H%M"))
            if self.numOfEpochs >= 5:
                self.checkPointsDir = os.path.join(self.model_weights_dir, 'weights_{}'.format(self.model_final_name))
                os.mkdir(self.checkPointsDir)
            else:
                self.checkPointsDir = self.model_weights_dir

        self.trainBatchSize, self.testBatchSize = 5000, 1000 #500, 500
        self.trainGen, self.testGen = self.createTrainAndTestGeneratros(datasetPath, self.trainBatchSize, self.testBatchSize, normalizationType='old')

        self.model.compile(
            optimizer=Adam(learning_rate=self.learningRate),
            loss=self.loss_dict,
            # metrics=[metrics.MeanSquaredError(name='mse'), metrics.MeanAbsoluteError(name='mae')])
            metrics=self.metric_dict)
    
    def createTrainAndTestGeneratros(self, datasetPath, trainBatchSize, testBatchSize, normalizationType='new'):
        inputImageShape = (480, 640, 3) 
        df = pd.read_pickle(datasetPath) 
        # randomize the data
        df = df.sample(frac=1, random_state=1)
        df.reset_index(drop=True, inplace=True)

        train_df = df.sample(frac=1, random_state=10)
        # test_df = df.drop(labels=train_df.index, axis=0)
        test_df = None

        train_df = df
        train_Xset, train_Yset = [train_df['markersData'].tolist(), train_df['vel'].tolist()], [train_df['positionControlPoints'].tolist(), train_df['yawControlPoints'].tolist()]
        # test_Xset, test_Yset = [test_df['markersData'].tolist(), test_df['vel'].tolist()], [test_df['positionControlPoints'].tolist(), test_df['yawControlPoints'].tolist()]
        statesProbList = train_df['statesProbList'] if self.apply_sampleWeight else None
        trainGenerator = MarkersAndTwistDataToBeizerDataGenerator(train_Xset, train_Yset, trainBatchSize, inputImageShape, self.config, statesProbList=statesProbList, normalizationType=normalizationType)
        # testGenerator = MarkersAndTwistDataToBeizerDataGenerator(test_Xset, test_Yset, testBatchSize, inputImageShape, self.config, normalizationType=normalizationType)
        testGenerator = None

        ## clearing RAM
        del df
        del train_df
        if test_df is not None:
            del test_df
        gc.collect()
        print('train and test generators has been created.')

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
                # self.model_final_name = 'config{}_ckpt_{}'.format(self.configNum, ''.join(ckpName))  + ''.join(final_name_splited[1:]) + '.h5'

        modelWeightsPath = os.path.join(self.model_weights_dir, 'wegihts_{}.h5'.format(self.model_final_name))
        modelHistoryPath = os.path.join(self.saveHistoryDir, 'history_{}.pkl'.format(self.model_final_name))


        #### callbacks definition
        callbacks = []
        # if not self.epochLearningRateRules is None:
        #     callbacks.append(tf.keras.callbacks.LearningRateScheduler(self.learningRateScheduler))
        
        ## checkpoints callback
        saveEveryEpochsNum = self.config.get('saveEveryEpochsNum', 10)
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(self.checkPointsDir, self.model_final_name),
            save_weights_only=True, save_freq= 'epoch')
        callbacks.append(model_checkpoint_callback)

        try:
            history = self.model.fit(
                x=self.trainGen, epochs=self.numOfEpochs, 
                validation_data=self.testGen, validation_steps=5, 
                callbacks=callbacks,
                verbose=1, workers=16, use_multiprocessing=True)
            with open(modelHistoryPath, 'wb') as file_pi:
                pickle.dump(history.history, file_pi) 
        except KeyboardInterrupt:
            print('KeyboardInterrupt, model weights were saved.')
        except Exception as e:
            print(e)
        finally:
            self.model.save_weights(modelWeightsPath)
            print('config{} is done.'.format(self.config['configNum']))

    def testModel(self, weights):
        self.model.load_weights(weights)
        _, testGen = self.createTrainAndTestGeneratros(1, 1)

        imageSet = [np.array2string(image_np[0, 0])[1:-1] for image_np in test_df['images'].tolist()]

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
    
    def evaluate(self, weights):
        self.model.load_weights(weights)
        result = self.model.evaluate(
                    x=self.testGen, 
                    verbose=1, workers=4, use_multiprocessing=True)
        positionCpMetrics = result[3]
        headingCpMetrics = result[4]
        print(result)
        print(positionCpMetrics)
        print(headingCpMetrics)
        print('Model X & {:.2E} & {:.2E} & {:.2E} & {:.2E} & {:.2E} & {:.2E}\\\\'.format(
            positionCpMetrics[0],
            headingCpMetrics[0],
            positionCpMetrics[1],
            headingCpMetrics[1],
            positionCpMetrics[2],
            headingCpMetrics[2]
            ))
        print('test deataset samples num {}'.format(len(self.testGen)))


def train(configs, startFromCheckpointDict=None):
    # training = Training(configs[6])
    # training.trainModel()

    ## set the training to 1200
    # print('changing cofings in a bad way!')
    for key in configs.keys():
        config = configs[key]
        # config['lossFunction'] = 'BezierLoss'
        # config['configNum'] = '{}_BeizerLoss'.format(config['configNum'])
        config['numOfEpochs'] = 500
        configs[key] = config 

    for key in configs.keys():
        training = Training(configs[key])
        training.trainModel(startFromCheckpointDict)

def test(configsRootDir):
    configs = ['config17']
    allConfigs = loadAllConfigs(configsRootDir, configs)
    training = Training(allConfigs)
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
    listOfConfigNums = ['config17'] #, 'config40', 'config40A', 'config41', 'config42', 'config43'] 
    # listOfConfigNums_colab0 = ['config17']
    # listOfConfigNums_colab1 = ['config15']

    # arg0 = sys.argv[1] if len(sys.argv) > 1 else ''
    # if arg0 == '':
    #     pass
    # elif arg0 == 'colab0':
    #     listOfConfigNums = listOfConfigNums_colab0
    #     print('colab0 is selected')
    # elif arg0 == 'colab1':
    #     listOfConfigNums = listOfConfigNums_colab1
    #     print('colab1 is selected')
    # elif arg0 == 'configs5':
    #     listOfConfigNums = ['config40', 'config41']
    #     print('working with configs5')
    # elif arg0 == 'existed':
    #     pass
    # else:
    #     listOfConfigNums = [arg0]
        
    print('listOfConfigNums:', listOfConfigNums)

    startFromCheckpoint = None # {
    #     'config17': '/home/majd/catkin_ws/src/basic_rl_agent/data/deep_learning/MarkersToBezierDataFolder/models_weights/wegihts_config17_BeizerLoss_imageToBezierData1_1800_20210905-1315.h5',
    #     'config61': '/home/majd/catkin_ws/src/basic_rl_agent/data/deep_learning/MarkersToBezierDataFolder/models_weights/wegihts_config61_imageToBezierData1_1800_20210906-1247.h5' 
    # }

    allConfigs = loadAllConfigs(configsRootDir, listOfConfigNums)
    train(allConfigs, startFromCheckpoint)

def evalute(configsRootDir):
    weights_path = "/home/majd/catkin_ws/src/basic_rl_agent/data/deep_learning/MarkersToBezierDataFolder/models_weights"
    # weights_filename = "wegihts_config17_BeizerLoss_imageToBezierData1_1800_20210905-1315.h5"
    # weights_filename = "wegihts_config61_DataWithRatio_imageBezierData1_183K_midPointData2_107k_20210909-1256_1800_20210909-0645.h5"
    weights_filename = "weights_config37_allData_imageBezierData1_midPointData_20210908-0018_1800_20210908-0029/weights_config37_allData_imageBezierData1_midPointData_20210908-0018_1800.h5"
    config = 'config37'
    allConfigs = loadAllConfigs(configsRootDir, [config])
    target_config = allConfigs[config]
    train = Training(target_config, evaluateOnly=True)
    train.evaluate(os.path.join(weights_path, weights_filename))

if __name__ == '__main__':
    trainOnConfigs(configsRootDir='/home/majd/catkin_ws/src/basic_rl_agent/scripts/learning/MarkersToBezierRegression/configs')

    # configs_file = '/home/majd/catkin_ws/src/basic_rl_agent/scripts/learning/MarkersToBezierRegression/configs/configs3.yaml'
    # configs = loadConfigsFromFile(configs_file)
    # train(configs)

    # test()
    # evalute(configsRootDir='/home/majd/catkin_ws/src/basic_rl_agent/scripts/learning/MarkersToBezierRegression/configs')