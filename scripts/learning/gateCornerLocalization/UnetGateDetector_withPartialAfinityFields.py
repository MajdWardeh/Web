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
        cornersOutput = layers.Conv2D(4, (1, 1), activation='linear', name='cornersOutput')(c9)
        pafsOutput = layers.Conv2D(8, (1, 1), activation='linear', name='pafsOutput')(c9)
        model = Model(inputs=inputs, outputs=[cornersOutput, pafsOutput])
        return model


class Training:

    def __init__(self):
        self.imageSize = (480, 640, 3)
        self.model = Unet(self.imageSize).getModel()
        self.model.summary()
        self.trainBatchSize, self.testBatchSize = 2, 1
        self.trainGen, self.testGen = self.createTrainAndTestGeneratros()

        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={'cornersOutput': 'mean_squared_error', 'pafsOutput': 'mean_squared_error'},
            # metrics=[metrics.MeanSquaredError(name='mse'), metrics.MeanAbsoluteError(name='mae')])
            metrics={'cornersOutput': metrics.MeanAbsoluteError(), 'pafsOutput': metrics.MeanAbsoluteError()} )
    
    def createTrainAndTestGeneratros(self):
        df = mergeDatasets('/home/majd/catkin_ws/src/basic_rl_agent/data/imageMarkersDataWithID')
        train_df = df.sample(frac=0.8, random_state=1)
        test_df = df.drop(labels=train_df.index, axis=0)
        train_Xset, train_Yset = train_df['images'].tolist(), train_df['markersArrays'].tolist()
        test_Xset, test_Yset = test_df['images'].tolist(), test_df['markersArrays'].tolist()
        trainGenerator = CornerPAFsDataGenerator(train_Xset, train_Yset, self.trainBatchSize, imageSize=self.imageSize, segma=7)
        testGenerator = CornerPAFsDataGenerator(test_Xset, test_Yset, self.testBatchSize, imageSize=self.imageSize, segma=7)
        return trainGenerator, testGenerator


    def trainModel(self):
        try:
            history = self.model.fit(
                x=self.trainGen, epochs=35, 
                validation_data=self.testGen, validation_steps=5, 
                # callbacks=[tensorboardCallback],
                verbose=1, workers=4, use_multiprocessing=True)
            with open('./trainHistoryDict/history_withPAFs_{}.pkl'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")), 'wb') as file_pi:
                pickle.dump(history.history, file_pi) 
        except KeyboardInterrupt:
            print('KeyboardInterrupt, model weights were saved.')
        finally:
            self.model.save_weights('./model_weights/weights_unet_scratch_withPAFs_{}.h5'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

    def testModel(self):
        def processCorners(conrers):
            # corners_hat[corners_hat < 0.3] = 0
            for i in range(3):
                corners_hat[:, :, i] += corners_hat[:, :, -1] 
            return (corners_hat*255).astype(np.uint8)

        def processPAFs(pafs):
            pafs_image = np.zeros(self.imageSize)
            zeros = np.zeros((self.imageSize[0], self.imageSize[1]))
            for j in range(4):
                paf = pafs[:, :, 2*j:2*j+2]
                pafs_image[:, :, 0] += (np.maximum(zeros, paf[:, :, 0]) * 128).astype(np.uint8)
                pafs_image[:, :, 1] += (np.maximum(zeros, -paf[:, :, 0]) * 128).astype(np.uint8)
                pafs_image[:, :, 2] += (np.maximum(zeros, paf[:, :, 1]) * 128).astype(np.uint8)
                pafs_image[:, :, 2] += (np.maximum(zeros, -paf[:, :, 1]) * 128).astype(np.uint8)
            return pafs_image

        self.model.load_weights('./model_weights/weights_unet_scratch_withPAFs_20210805-141359.h5')
        for k in range(self.testGen.__len__()):
            x, y = self.testGen.__getitem__(k)
            y_hat = self.model(x, training=False)
            corners_gt, pafs_gt = y[0][0], y[1][0]
            corners_hat, pafs_hat = y_hat[0][0].numpy(), y_hat[1][0].numpy()

            x = (x[0] * 128).astype(np.uint8)
            corners_hat = processCorners(corners_hat)
            corners_gt = processCorners(corners_gt)
            pafs_hat = processPAFs(pafs_hat)
            pafs_gt = processPAFs(pafs_gt)
            # y_hat_gray = (np.sum(corners_hat, axis=2) * 255).astype(np.uint8)
            # y_hat_gray = y_hat_gray[:, :,  np.newaxis]
            # y_hat_rgb = x.copy()
            # y_hat_rgb[(y_hat_gray != 0).reshape(480, 640)] = y_hat_gray[(y_hat_gray!=0).reshape(480, 640)]

            cv2.imshow('image', x)
            cv2.imshow('corners_hat', corners_hat)
            cv2.imshow('corners_gt', corners_gt)
            cv2.imshow('pafs_hat', pafs_hat)
            cv2.imshow('pafs_gt', pafs_gt)
            if cv2.waitKey(1000) & 0xFF == ord('q'): 
                break


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
        

def main():
    training = Training()
    training.trainModel()
    # training.testModel()


    

if __name__ == '__main__':
    main()