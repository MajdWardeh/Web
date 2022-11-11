import sys
# from tensorflow.python.keras.backend import square
# from tensorflow.python.keras.metrics import Precision

# from tensorflow.python.ops.gen_math_ops import Square
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import datetime
import numpy as np
import numpy.linalg as la
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
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.losses import Loss, MeanAbsoluteError, MeanSquaredError

from tensorflow.python.compiler.tensorrt import trt_convert as trt

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
        self.imageSize = (240, 320, 3)
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
        df = mergeDatasets('/home/majd/catkin_ws/src/basic_rl_agent/data/imageMarkersDataWithDronePoses')
        train_df = df.sample(frac=0.8, random_state=1)
        test_df = df.drop(labels=train_df.index, axis=0)
        train_Xset, train_Yset = train_df['images'].tolist(), train_df['markersArrays'].tolist()
        test_Xset, test_Yset = test_df['images'].tolist(), test_df['markersArrays'].tolist()
        print('# of training images {}, # of testing images: {}'.format(len(train_df['images']), len(test_df['images'])))
        trainGenerator = CornersDataGeneratorWithZ(train_Xset, train_Yset, self.trainBatchSize, imageSize=self.imageSize, segma=5)
        testGenerator = CornersDataGeneratorWithZ(test_Xset, test_Yset, self.testBatchSize, imageSize=self.imageSize, segma=5)
        return trainGenerator, testGenerator


    def trainModel(self):
        try:
            history = self.model.fit(
                x=self.trainGen, epochs=15, 
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
        self.model.load_weights(os.path.join(self.model_weights_dir, 'weights_unet_scratch_cornersWtihZ_20220816-224802.h5'))
        for k in range(self.testGen.__len__()):
            x, y = self.testGen.__getitem__(k)
            print(k)
            y_hat = self.model(x, training=False)
            corners_hat, z_hat = y_hat[0][0], y_hat[1][0]

            x = (x[0] * 255).astype(np.uint8)
            corners_hat = y[0][0]
            # corners_hat = corners_hat.numpy()
            print(x.shape)
            print(corners_hat.shape)
            print(corners_hat.max())
            
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
            cv2.imshow('allCorners', allCorners)
            # cv2.imshow('z', allZs)
            if cv2.waitKey(0) == ord('q'):
                break
    
    def save_mode(self, path):
        self.model.save(path)

    def optimizeWithTensorRT(self, original_model_path, optimized_model_path):
        conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
                                        precision_mode=trt.TrtPrecisionMode.FP32)

        converter = trt.TrtGraphConverterV2(
            input_saved_model_dir=original_model_path,
            conversion_params=conversion_params)
        converter.convert()
        converter.save(optimized_model_path)

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
        l2ErrorList = []
        inferenceTimeList = []
        for k in range(self.testGen.__len__())[:]: # take only one image
            x, y = self.testGen.__getitem__(k)
            ts = time.time()
            y_hat = self.model(x, training=False)
            inferenceTimeList.append(time.time()-ts)

            z_gt = y[1][0]
            corners_hat, z_hat = y_hat[0][0], y_hat[1][0]
            # corners_hat, z_hat = y_hat[0][0].numpy(), y_hat[1][0].numpy()
            # for i in range(4):
            #     corner = corners_hat[:, :, i]
                # corner_filtered = cv2.GaussianBlur(corner, ksize=(0, 0), sigmaX=5)
                # corners_hat[:, :, i] = corner_filtered
            image = (x[0] * 255).astype(np.uint8)
            gateEstimatedCornerLocations = self.process_corners(corners_hat, image) 
            gateMarkersData = self.testGen.getMarkersData(k)[0]
            gateMarkersData = np.rint(gateMarkersData).astype(np.int)

            if gateEstimatedCornerLocations.shape != (4, 2):
                print("got wronge estimation ", gateEstimatedCornerLocations.shape)
            else:
                error = gateMarkersData[:, :-1] - gateEstimatedCornerLocations
                l2_error = la.norm(error, axis=-1)
                # squaredError = np.square(error)
                l2ErrorList.append(l2_error)
            
            # for marker in gateMarkersData:
            #     image = cv2.circle(image, (marker[0], marker[1]), 2, (0, 255, 0), -1)
            # cv2.imshow('image', image)
            # cv2.waitKey(0)

        inferenceTimeArray = np.array(inferenceTimeList)
        print('meanInferenceTime:', inferenceTimeArray.mean(), 1/inferenceTimeArray.mean())

        l2ErrorArray = np.array(l2ErrorList)
        print('Error analysis:')
        min_error = l2ErrorArray.min()
        max_error = l2ErrorArray.max()
        print("mean: {}, Min: {}, Max: {}, shape: {}".format(l2ErrorArray.mean(), min_error, max_error, l2ErrorArray.shape))

        all_values = np.unique(l2ErrorArray)
        errorCountDect = {}
        l2ErrorFlattened = l2ErrorArray.flatten()
        for val in all_values:
            c = len(l2ErrorFlattened) - np.count_nonzero(l2ErrorFlattened-val)
            errorCountDect[val] = c
        print(len(all_values))
        print(all_values)
        
        print(errorCountDect)

        fig = plt.figure()
        d = np.diff(np.unique(l2ErrorFlattened)).min()
        left_of_first_bin = l2ErrorFlattened.min() - float(d)/2
        right_of_last_bin = l2ErrorFlattened.max() + float(d)/2
        plt.hist(l2ErrorFlattened, np.arange(left_of_first_bin, right_of_last_bin + d, d))
        # plt.hist(l2ErrorArray.flatten(), bins=len(all_values))
        # plt.hist(l2ErrorFlattened, range=(min_error, max_error))
        # plt.title('The histogram of the L2 error between the ground-truth and predicted corners')
        plt.ylabel('Corners Count')
        plt.xlabel('L2 error [pixels]')
        plt.show()


        # errors_dict = {}
        # for e in range(min_error, max_error + 1):

        #     e_count = len(L2ErrorArray) - np.count_nonzero(L2ErrorArray - e)
        #     errors_dict[e] = e_count
        # print(errors_dict)


            # x = (x[0] * 255).astype(np.uint8)
            # allCorners = (np.sum(corners_hat, axis=2) * 255).astype(np.uint8)
            # imageWithCorners = x.copy().astype(np.uint16)
            # for c in range(3):
            #     imageWithCorners[:, :, c] = np.minimum(imageWithCorners[:, :, c] + allCorners, 255)
            # imageWithCorners = imageWithCorners.astype(np.uint8)

            # allCorners_hat = np.zeros(shape=(self.imageSize[0], self.imageSize[1]), dtype=np.uint8)
            # for i in range(4):
            #     # nonZermakre.unravel_index(maxZi, z_hat[:, :, i].shape)
            #     allCorners_hat[idx] = 255

            # allZs = np.sum(z_hat, axis=2)
            # z_hat0 = z_hat[:, :, 0]
            # # print(z_hat0.shape)
            # indices = z_hat0 > 0.5
            # z_gt0 = z_gt[:, :, 0]
            # # print((z_gt0[indices], z_hat0[indices]))
            # mse = np.mean(np.square(z_gt0[indices] -z_hat0[indices]), axis=-1)
            # print(mse)
            # print(np.min(z_hat0), np.max(z_hat0), z_hat0[z_hat0>0.1].shape)

            # cv2.imshow('image', x)
            # # self.process_Zimage(allZs)
            # cv2.imshow('allCorners_hat', allCorners_hat)
            # cv2.imshow('imageWithCorners', imageWithCorners)
            # cv2.imshow('z', allZs)
            # cv2.waitKey(0)
    
    def process_corners(self, all_corner, image):
        cornerRGB = image
        gateCornersLocations = []
        for cornerID in range(4):
            corner = all_corner[:, :, cornerID]
            corner = tf.reshape(corner, (1, corner.shape[0], corner.shape[1], 1))
            # corner = corner[np.newaxis, :, :, np.newaxis]

            # max_pooled_in_tensor = tf.nn.pool(corner, window_shape=(5, 5), pooling_type='MAX', padding='SAME')
            max_pooled_in_tensor = tf.nn.max_pool(corner, ksize=5, strides=1, padding='SAME')
            maxima = tf.where(tf.math.logical_and(tf.equal(corner, max_pooled_in_tensor), corner > 0.85) )

            maxima = maxima.numpy()
            for i in range(maxima.shape[0]):
                cornerRGB = cv2.circle(cornerRGB, (maxima[i, 2], maxima[i, 1]), 2, (255, 0, 0), -1)
                gateCornersLocations.append([maxima[i, 2], maxima[i, 1]]) # on the first element is on the x axis and the second on the y.

        # cv2.imshow('corner', cornerRGB)
        # cv2.waitKey(600)
        return np.array(gateCornersLocations)

            


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

    def opencv_test(self):
        pass
        

def main():

    model_weights_dir = '/home/majd/catkin_ws/src/basic_rl_agent/data2/flightgoggles/deep_learning/cornersDetector/model_weights'
    model_history_dir = '/home/majd/catkin_ws/src/basic_rl_agent/data2/flightgoggles/deep_learning/cornersDetector/trainHistoryDict'
    training = Training(model_weights_dir, model_history_dir)
    # training.save_mode('/home/majd/catkin_ws/src/basic_rl_agent/data/deep_learning/cornersDetector/tensorrt/original_model')
    # training.optimizeWithTensorRT(
    #     original_model_path='/home/majd/catkin_ws/src/basic_rl_agent/data/deep_learning/cornersDetector/tensorrt/original_model',
    #     optimized_model_path='/home/majd/catkin_ws/src/basic_rl_agent/data/deep_learning/cornersDetector/tensorrt/optimized_model_FP32'
    # )

    # training.trainModel() # we did in 20220815
    training.testModel()
    # training.testModelWithControus()
    # training.opencv_test()


    

if __name__ == '__main__':
    main()