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
from imageMarkersDataGenerator import CornerPAFsDataGenerator
from imageMarkersDataGeneratorWithZ import CornersDataGeneratorWithZ
from imageMarkersDatasetsMerging import mergeDatasets

class Unet:

    def __init__(self, inputShape=(240, 320, 3,)):
        self.inputShape = inputShape
        self.model = self._create_unet()

    def getModel(self):
        return self.model

    def _conv2d_block(self, input_tensor, n_filters, kernel_size=3):
        x = input_tensor
        for i in range(2):
            x = layers.Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
                    kernel_initializer = 'he_normal', padding = 'same')(x)
            x = layers.Activation('relu')(x)
        return x

    def _encoder_block(self, inputs, n_filters=64, pool_size=(2,2), dropout=0.3):
        f = self._conv2d_block(inputs, n_filters=n_filters)
        p = layers.MaxPooling2D(pool_size=pool_size)(f)
        p = layers.Dropout(dropout)(p)
        return f, p 

    def _encoder(self, inputs):
        '''
        Returns:
            p4 - the output maxpooled features of the last encoder block
            (f1, f2, f3, f4) - the output features of all the encoder blocks
        '''
        f1, p1 = self._encoder_block(inputs, n_filters=64, pool_size=(2,2), dropout=0.3)
        f2, p2 = self._encoder_block(p1, n_filters=128, pool_size=(2,2), dropout=0.3)
        f3, p3 = self._encoder_block(p2, n_filters=256, pool_size=(2,2), dropout=0.3)
        f4, p4 = self._encoder_block(p3, n_filters=512, pool_size=(2,2), dropout=0.3)
        return p4, (f1, f2, f3, f4) 

    def _bottleneck(self, inputs):
        bottle_neck = self._conv2d_block(inputs, n_filters=1024)
        return bottle_neck

    def _decoder_block(self, inputs, conv_output, n_filters, kernel_size=3, strides=3, dropout=0.3):
        u = layers.Conv2DTranspose(n_filters, kernel_size, strides=strides, padding='same')(inputs)
        c = layers.concatenate([u, conv_output])
        c = layers.Dropout(dropout)(c)
        c = self._conv2d_block(c, n_filters, kernel_size=3)
        return c

    def _decoder(self, inputs, convs):
        f1, f2, f3, f4 = convs
        c6 = self._decoder_block(inputs, f4, n_filters=512, kernel_size=(3,3), strides=(2,2), dropout=0.3)
        c7 = self._decoder_block(c6, f3, n_filters=256, kernel_size=(3,3), strides=(2,2), dropout=0.3)
        c8 = self._decoder_block(c7, f2, n_filters=128, kernel_size=(3,3), strides=(2,2), dropout=0.3)
        c9 = self._decoder_block(c8, f1, n_filters=64, kernel_size=(3,3), strides=(2,2), dropout=0.3)
        return c9
    
    def _create_unet(self):
        # specify the input shape
        inputs = layers.Input(shape=(self.inputShape[0], self.inputShape[1], 3,))
        # feed the inputs to the encoder
        encoder_output, convs = self._encoder(inputs)
        # feed the encoder output to the bottleneck
        bottle_neck = self._bottleneck(encoder_output)
        # feed the bottleneck and encoder block outputs to the decoder
        c9 = self._decoder(bottle_neck, convs)
        # c9 is the last layer before the output layer
        # TODO change output channels later
        cornersOutputs = layers.Conv2D(4, (1, 1), activation='linear', name='cornersOut')(c9)
        zOutputs = layers.Conv2D(4, (1, 1), activation='linear', name='zOut')(c9)
        
        model = Model(inputs=inputs, outputs=[cornersOutputs, zOutputs])
        return model


class Training:

    def __init__(self, model_weights_dir, saveHistoryDir):
        self.imageSize = (480, 640, 3)
        self.model = Unet(self.imageSize).getModel()
        self.trainBatchSize, self.testBatchSize = 2, 1
        self.trainGen, self.testGen = self.createTrainAndTestGeneratros()
        self.model_weights_dir = model_weights_dir
        self.saveHistoryDir = saveHistoryDir

        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={'cornersOut': 'mean_squared_error', 'zOut': 'mean_squared_error'},
            # metrics=[metrics.MeanSquaredError(name='mse'), metrics.MeanAbsoluteError(name='mae')])
            metrics={'cornersOut': metrics.MeanAbsoluteError(), 'zOut': metrics.MeanAbsoluteError()} )
    
    def createTrainAndTestGeneratros(self):
        df = mergeDatasets('/home/majd/catkin_ws/src/basic_rl_agent/data/imageMarkersDataWithID')
        train_df = df.sample(frac=0.8, random_state=1)
        test_df = df.drop(labels=train_df.index, axis=0)
        train_Xset, train_Yset = train_df['images'].tolist(), train_df['markersArrays'].tolist()
        test_Xset, test_Yset = test_df['images'].tolist(), test_df['markersArrays'].tolist()
        trainGenerator = CornersDataGeneratorWithZ(train_Xset, train_Yset, self.trainBatchSize, imageSize=self.imageSize, segma=5)
        testGenerator = CornersDataGeneratorWithZ(test_Xset, test_Yset, self.testBatchSize, imageSize=self.imageSize, segma=5)
        return trainGenerator, testGenerator


    def trainModel(self):
        try:
            history = self.model.fit(
                x=self.trainGen, epochs=30, 
                validation_data=self.testGen, validation_steps=5, 
                # callbacks=[tensorboardCallback],
                verbose=1, workers=4, use_multiprocessing=True)
            with open(os.path.join(self.saveHistoryDir, 'history_unet_scratch_cornersWtihZ_{}.pkl'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))), 'wb') as file_pi:
                pickle.dump(history.history, file_pi) 
        except KeyboardInterrupt:
            print('KeyboardInterrupt, model weights were saved.')
        finally:
            self.model.save_weights(os.path.join(self.model_weights_dir, 'weights_unet_scratch_cornersWtihZ_{}.h5'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))))

    def testModel(self):
        self.model.load_weights(os.path.join(self.model_weights_dir, 'weights_unet_scratch_cornersWtihZ_20210814-101152.h5'))
        for k in range(self.testGen.__len__()):
            x, y = self.testGen.__getitem__(k)
            y_hat = self.model(x, training=False)
            corners_hat, z_hat = y_hat[0][0], y_hat[1][0]

            x = (x[0] * 255).astype(np.uint8)
            print(x.shape)
            allCorners = (np.sum(corners_hat, axis=2) * 255).astype(np.uint8)
            imageWithCorners = x.copy().astype(np.uint16)
            for c in range(3):
                imageWithCorners[:, :, c] = np.minimum(imageWithCorners[:, :, c] + allCorners, 255)
            imageWithCorners = imageWithCorners.astype(np.uint8)
            for i in range(4):
                nonZeros = np.argwhere(z_hat[:, :, i] > 0.2)
                print(i, nonZeros.shape, np.max(z_hat[:, :, i]))

            allZs = np.sum(z_hat, axis=2)
            cv2.imshow('image', x)
            cv2.imshow('imageWithCorners', imageWithCorners)
            cv2.imshow('z', allZs)
            cv2.waitKey(1500)

    def testModelInferenceTime(self):
        self.model.load_weights('./model_weights/weights_unet_scratch_20210802-082621.h5')
        print(self.testGen.__len__())
        t_sum = 0
        for k in range(self.testGen.__len__()):
            x, y = self.testGen.__getitem__(k)
            ts = time.time()
            y_hat = self.model(x, training=False)
            te = time.time()
            t_sum += te-ts
        print(self.testGen.__len__()/t_sum)
        
    def testModelWithControus(self):
        self.model.load_weights(os.path.join(self.model_weights_dir, 'weights_unet_scratch_cornersWtihZ_20210814-101152.h5'))
        for k in range(self.testGen.__len__())[:1]: # take only one image
            x, y = self.testGen.__getitem__(k)
            y_hat = self.model(x, training=False)
            corners_hat, z_hat = y_hat[0][0], y_hat[1][0]

            x = (x[0] * 255).astype(np.uint8)
            allCorners = (np.sum(corners_hat, axis=2) * 255).astype(np.uint8)
            imageWithCorners = x.copy().astype(np.uint16)
            for c in range(3):
                imageWithCorners[:, :, c] = np.minimum(imageWithCorners[:, :, c] + allCorners, 255)
            imageWithCorners = imageWithCorners.astype(np.uint8)

            allCorners_hat = np.zeros(shape=(self.imageSize[0], self.imageSize[1]), dtype=np.uint8)
            for i in range(4):
                # nonZeros = np.argwhere(z_hat[:, :, i] > 0.2)
                # print(i, nonZeros.shape, np.max(z_hat[:, :, i]))
                maxZi = np.argmax(z_hat[:, :, i])
                idx = np.unravel_index(maxZi, z_hat[:, :, i].shape)
                print(idx)
                allCorners_hat[idx] = 255

            allZs = np.sum(z_hat, axis=2)
            cv2.imshow('image', x)
            cv2.imshow('allCorners_hat', allCorners_hat)
            cv2.waitKey(0)

            # self.process_Zimage(allZs)

            # cv2.imshow('image', x)
            # cv2.imshow('imageWithCorners', imageWithCorners)
            # cv2.imshow('z', allZs)
            # cv2.waitKey(0)

    def process_Zimage(self, image):
        maxZ = np.max(image)
        image = image * 255.0/maxZ
        imgray = image.astype(np.uint8)
        ret, thresh = cv2.threshold(imgray, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(imgray, contours, -1, (0,255,0), 3)
        cv2.imshow('thresh', thresh)
        cv2.imshow('Zimage', imgray)
        cv2.waitKey(0)

def main():
    model_weights_dir = '/home/majd/catkin_ws/src/basic_rl_agent/data/deep_learning/cornersDetector/model_weights'
    model_history_dir = '/home/majd/catkin_ws/src/basic_rl_agent/data/deep_learning/cornersDetector/trainHistoryDict'
    training = Training(model_weights_dir, model_history_dir)
    # training.trainModel()
    # training.testModel()
    training.testModelWithControus()


    

if __name__ == '__main__':
    main()