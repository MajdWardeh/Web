import sys

from tensorflow.python.keras import activations
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
from MarkersToBezierGenerator import MarkersDataToBeizerDataGenerator

from Bezier_untils import bezier4thOrder

class Network:

    def __init__(self, dropoutRate=0.3):
        self.dropoutRate = dropoutRate
        self.model = self._createModel()

    def getModel(self):
        return self.model

    def _createModel(self):
        inputLayer = layers.Input(shape=(12, ))
        x = layers.Dense(150, activation='relu')(inputLayer)
        x = layers.Dropout(self.dropoutRate)(x)
        x = layers.Dense(80, activation='relu')(x)
        x = layers.Dropout(self.dropoutRate)(x)
        x = layers.Dense(80, activation='relu')(x)
        outputLayer = layers.Dense(15, activation='linear')(x)
        model = Model(inputs=inputLayer, outputs=outputLayer)
        return model

class Training:

    def __init__(self):
        self.model = Network().getModel()
        # self.model.summary()
        self.trainBatchSize, self.testBatchSize = 1000, 1000 #500, 500
        self.trainGen, self.testGen = self.createTrainAndTestGeneratros(self.trainBatchSize, self.testBatchSize)
        self.model_weights_dir = '/home/majd/catkin_ws/src/basic_rl_agent/data/deep_learning/MarkersToBezierDataFolder/models_weights'
        self.saveHistoryDir = '/home/majd/catkin_ws/src/basic_rl_agent/data/deep_learning/MarkersToBezierDataFolder/trainHistoryDict'

        self.model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss='mean_squared_error',
            # metrics=[metrics.MeanSquaredError(name='mse'), metrics.MeanAbsoluteError(name='mae')])
            metrics=[metrics.MeanAbsoluteError()])
    
    def createTrainAndTestGeneratros(self, trainBatchSize, testBatchSize):
        allDataFileWithMarkers = '/home/majd/catkin_ws/src/basic_rl_agent/data/debugging_data2/allDataWithMarkers.pkl'
        inputImageShape = (480, 640, 3) 
        df = pd.read_pickle(allDataFileWithMarkers) 
        # randomize the data
        df = df.sample(frac=1, random_state=1)
        df.reset_index(drop=True, inplace=True)

        self.train_df = df.sample(frac=0.8, random_state=10)
        test_df = df.drop(labels=self.train_df.index, axis=0)
        train_Xset, train_Yset = self.train_df['markersData'].tolist(), [self.train_df['positionControlPoints'].tolist(), self.train_df['yawControlPoints'].tolist()]
        test_Xset, test_Yset = test_df['markersData'].tolist(), [test_df['positionControlPoints'].tolist(), test_df['yawControlPoints'].tolist()]
        trainGenerator = MarkersDataToBeizerDataGenerator(train_Xset, train_Yset, trainBatchSize, inputImageShape)
        testGenerator = MarkersDataToBeizerDataGenerator(test_Xset, test_Yset, testBatchSize, inputImageShape)
        return trainGenerator, testGenerator

    def trainModel(self):
        #TODO if training was less than 5 minutes, don't save weights.
        try:
            history = self.model.fit(
                x=self.trainGen, epochs=10000, 
                validation_data=self.testGen, validation_steps=5, 
                # callbacks=[tensorboardCallback],
                verbose=1, workers=4, use_multiprocessing=True)
            with open(os.path.join(self.saveHistoryDir, 'weights_MarkersToBeizer_FC_scratc_{}.pkl'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))), 'wb') as file_pi:
                pickle.dump(history.history, file_pi) 
        except KeyboardInterrupt:
            print('KeyboardInterrupt, model weights were saved.')
        finally:
            self.model.save_weights(os.path.join(self.model_weights_dir, 'weights_MarkersToBeizer_FC_scratch_{}.h5'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))))

    def testModel(self):
        self.model.load_weights(os.path.join(self.model_weights_dir, 'weights_MarkersToBeizer_FC_scratch_20210809-031406.h5'))
        _, testGen = self.createTrainAndTestGeneratros(1, 1)
        imageSet = [np.array2string(image_np[0, 0])[1:-1] for image_np in test_df['images'].tolist()]
        fig = plt.figure(figsize=plt.figaspect(1/3.0))
        ax = [0] * 2
        ax[0] = fig.add_subplot(1, 3, 1, projection='3d')
        ax[1] = fig.add_subplot(1, 3, 2, projection='3d')
        ax2   = fig.add_subplot(1, 3, 3)

        # ax = fig.gca(projection='3d')
        # ax.set_aspect('equal')
        plt.ion()
        plt.show()
        for k in range(1000, testGen.__len__())[:]:
            print(k)
            x, y = testGen.__getitem__(k)
            y_hat = self.model(x, training=False)
            y = y[0]
            y_hat = y_hat[0].numpy()
            print(np.mean(np.abs(y-y_hat)))
            imageName = imageSet[k]
            image = cv2.imread(imageName)
            ax2.imshow(image)
            self.plotBezier(ax, y, y_hat)

    def plotBezier(self, axes, cp, cp_hat):
        def process(cp):
            cp = cp.reshape(5, 3)
            cp = cp.T
            acc = 100
            t_space = np.linspace(0, 1, acc)
            Ps = []
            for ti in t_space:
                P = bezier4thOrder(cp, ti) 
                Ps.append(P)
            Ps = np.array(Ps)
            return Ps

        Ps = process(cp)
        Ps_hat = process(cp_hat)

        for ax in axes:
            ax.clear()
            axLim = 2
            ax.set_xlim(-axLim, axLim)
            ax.set_ylim(-axLim, axLim)
            ax.set_zlim(-axLim, axLim)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.azim = 180
            ax.dist = 8
            #ax.elev = 45
            ax.plot3D(Ps[:, 0], Ps[:, 1], Ps[:, 2], 'r')
            ax.plot3D(Ps_hat[:, 0], Ps_hat[:, 1], Ps_hat[:, 2], 'b')
        axes[0].elev = 0
        axes[1].elev = 90

        plt.draw()
        plt.pause(1.5)


def main():
    training = Training()
    training.trainModel()
    # training.testModel()


    

if __name__ == '__main__':
    main()