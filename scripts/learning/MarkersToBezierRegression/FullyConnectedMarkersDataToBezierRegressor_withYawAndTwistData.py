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
from MarkersToBezierGenerator import MarkersAndTwistDataToBeizerDataGenerator

from Bezier_untils import BezierVisulizer, bezier4thOrder

class Network:

    def __init__(self, dropoutRate=0.3):
        self.dropoutRate = dropoutRate
        self.model = self._createModel()

    def getModel(self):
        return self.model

    def _createModel(self):
        markersDataInput = layers.Input(shape=(12, ))
        twistDataInput = layers.Input(shape=(8, ))
        x = layers.concatenate([markersDataInput, twistDataInput], axis=1)
        x = layers.Dense(150, activation='relu')(x)
        x = layers.Dropout(self.dropoutRate)(x)
        x = layers.Dense(80, activation='relu')(x)
        x = layers.Dropout(self.dropoutRate)(x)
        x = layers.Dense(80, activation='relu')(x)
        positionOutput = layers.Dense(15, activation='linear', name='positionOutput')(x)
        yawOutput = layers.Dense(3, activation='linear', name='yawOutput')(x)

        model = Model(inputs=[markersDataInput, twistDataInput], outputs=[positionOutput, yawOutput])
        return model

class Training:

    def __init__(self):
        self.model = Network().getModel()
        self.model.summary()
        self.trainBatchSize, self.testBatchSize = 1000, 1000 #500, 500
        self.trainGen, self.testGen = self.createTrainAndTestGeneratros(self.trainBatchSize, self.testBatchSize)
        self.model_weights_dir = '/home/majd/catkin_ws/src/basic_rl_agent/data/deep_learning/MarkersToBezierDataFolder/models_weights'
        self.saveHistoryDir = '/home/majd/catkin_ws/src/basic_rl_agent/data/deep_learning/MarkersToBezierDataFolder/trainHistoryDict'

        self.model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss={'positionOutput': 'mean_squared_error', 'yawOutput': 'mean_squared_error'},
            # metrics=[metrics.MeanSquaredError(name='mse'), metrics.MeanAbsoluteError(name='mae')])
            metrics={'positionOutput': metrics.MeanAbsoluteError(), 'yawOutput':metrics.MeanAbsoluteError()})
    
    def createTrainAndTestGeneratros(self, trainBatchSize, testBatchSize):
        allDataFileWithMarkers = '/home/majd/catkin_ws/src/basic_rl_agent/data/debugging_data2/allDataWithMarkers.pkl'
        inputImageShape = (480, 640, 3) 
        df = pd.read_pickle(allDataFileWithMarkers) 
        # randomize the data
        df = df.sample(frac=1, random_state=1)
        df.reset_index(drop=True, inplace=True)

        self.train_df = df.sample(frac=0.8, random_state=10)
        self.test_df = df.drop(labels=self.train_df.index, axis=0)
        train_Xset, train_Yset = [self.train_df['markersData'].tolist(), self.train_df['vel'].tolist()], [self.train_df['positionControlPoints'].tolist(), self.train_df['yawControlPoints'].tolist()]
        test_Xset, test_Yset = [self.test_df['markersData'].tolist(), self.test_df['vel'].tolist()], [self.test_df['positionControlPoints'].tolist(), self.test_df['yawControlPoints'].tolist()]
        trainGenerator = MarkersAndTwistDataToBeizerDataGenerator(train_Xset, train_Yset, trainBatchSize, inputImageShape)
        testGenerator = MarkersAndTwistDataToBeizerDataGenerator(test_Xset, test_Yset, testBatchSize, inputImageShape)
        return trainGenerator, testGenerator

    def trainModel(self):
        #TODO if training was less than 5 minutes, don't save weights.
        try:
            history = self.model.fit(
                x=self.trainGen, epochs=800, 
                validation_data=self.testGen, validation_steps=5, 
                # callbacks=[tensorboardCallback],
                verbose=1, workers=4, use_multiprocessing=True)
            with open(os.path.join(self.saveHistoryDir, 'history_MarkersToBeizer_FC_scratch_withYawAndTwistData_{}.pkl'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))), 'wb') as file_pi:
                pickle.dump(history.history, file_pi) 
        except KeyboardInterrupt:
            print('KeyboardInterrupt, model weights were saved.')
        finally:
            self.model.save_weights(os.path.join(self.model_weights_dir, 'weights_MarkersToBeizer_FC_scratch_withYawAndTwistData_{}.h5'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))))

    def testModel(self):
        self.model.load_weights(os.path.join(self.model_weights_dir, 'weights_MarkersToBeizer_FC_scratch_withYawAndTwistData_20210809-193637.h5'))
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


def main():
    training = Training()
    training.trainModel()
    # training.testModel()


    

if __name__ == '__main__':
    main()