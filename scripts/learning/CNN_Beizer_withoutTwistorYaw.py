import sys

from tensorflow.python.keras import models

sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import datetime
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import cv2
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Input, layers, Model, backend as k
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.metrics as metrics
from tensorflow.keras.utils import Sequence
from keras.callbacks import TensorBoard

from tensorflow.keras.applications.inception_v3 import InceptionV3

from Bezier_untils import bezier4thOrder

class TensorBoardExtended(TensorBoard):
    """
    Extended Tensorboard log that allows to add text

    By default logs:
    - host
    - gpus available

    Parameters
    -------------
    text_dict_to_log : dict
        Dictionary with key, value string that will be logged with Tensorboard
    kwargs : dict
        All the other parameters that are fed to Tensorboard
    """
    def __init__(self, text_dict_to_log=None, **kwargs):
        super().__init__(**kwargs)
        self.text_dict_to_log = text_dict_to_log

    def on_train_begin(self, logs=None):
        # pylint: disable= E1101
        super().on_train_begin(logs=logs)
        #     writer = self._get_writer('train')
        writer = self._train_writer
        with writer.as_default():
            for key, value in self.text_dict_to_log.items():
                tf.summary.text(key, tf.convert_to_tensor(value), step=0)

def preprocessAllData(directory):
    df = pd.read_pickle(directory)

    # process images: removing the list
    imagesList = df['images'].tolist()
    imagesList = [image[0][0] for image in imagesList]
    print(imagesList[0])
    df.drop('images', axis = 1, inplace = True)
    df['images'] = imagesList

    # process positionControlPoints: remove a0=(0, 0, 0) from the np arrays.
    pcps = df['positionControlPoints'].tolist()
    pcps = [p[1:] for p in pcps]
    df.drop('positionControlPoints', axis = 1, inplace = True)
    df['positionControlPoints'] = pcps

    # process yawControlPoints: remove a0=(0, 0, 0) from the np arrays.
    yawControlPoints = df['yawControlPoints']
    yawControlPoints = [p[1:] for p in yawControlPoints]
    df.drop('yawControlPoints', axis=1, inplace=True)
    df['yawControlPoints'] = yawControlPoints

    return df

class DataGenerator(Sequence):

    def __init__(self, x_set, y_set, batch_size, twist_data_len):
        self.images_set = x_set[0]
        self.twist_set = x_set[1]
        self.y_position = y_set[0]
        self.y_yaw = y_set[1]
        self.batch_size = batch_size
        self.len = math.ceil(len(self.images_set) / self.batch_size) 
        self.TwistDataLength = twist_data_len

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # images_batch = np.zeros((self.batch_size, 100, 200, 3), dtype='float32')
        # y_batch = np.zeros((self.batch_size, 12), dtype='float32')
        images_batch = []
        twist_batch = []
        y_batch = [] # shape: (bs, 12) + (bs, 2) = (bs, 14)

        # fill up the batch
        for row in range(min(self.batch_size, len(self.images_set)-index*self.batch_size)):
            image = cv2.imread(self.images_set[index*self.batch_size + row])
            if image is None:
                continue
            image = cv2.resize(image, (320, 240)) # output shape is (240, 320, 3)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32)
            images_batch.append(image)

            # twist_data = self.twist_set[index*self.batch_size + row]
            # twist_data = twist_data[-self.TwistDataLength:, :]
            # twist_batch.append(twist_data)

            y_position_data = np.array(self.y_position[index*self.batch_size + row][:]).reshape((12, ))
            # y_yaw_data = np.array(self.y_yaw[index*self.batch_size + row][:]).reshape((2, )) 
            # y_batch.append(np.concatenate([y_position_data, y_yaw_data], axis=0))
            y_batch.append(y_position_data)

        images_batch = np.array(images_batch)
        twist_batch = np.array(twist_batch)
        y_batch = np.array(y_batch)
        # Normalize inputs
        images_batch = images_batch/255.
        # return ([images_batch, twist_batch], y_batch)
        return (images_batch, y_batch)

class Trainer:

    def _getInceptionModel(self):
        local_weights_file = '../inception_weights/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
        pre_trained_model = InceptionV3(input_shape = (240, 320, 3), 
                                        include_top = False, 
                                        weights = None)
        pre_trained_model.load_weights(local_weights_file)

        for layer in pre_trained_model.layers:
            layer.trainable = True

        last_layer = pre_trained_model.get_layer('mixed7')
        # print('last layer output shape: ', last_layer.output_shape)
        last_output = last_layer.output


        x = layers.Flatten()(last_output)

        # TwistInputLayer = Input(shape=(10, 4))
        # twistFlatten = layers.Flatten()(TwistInputLayer)
        # x = layers.concatenate([x, twistFlatten])

        # Add a fully connected layer with 1,024 hidden units and ReLU activation
        x = layers.Dense(1024, activation='relu')(x)
        # Add a dropout rate of 0.2
        x = layers.Dropout(0.2)(x)  
        x = layers.Dense(512, activation='relu')(x)                
        # output layer:
        x = layers.Dense(12, activation=None)(x) 
        # model = Model( [pre_trained_model.input, TwistInputLayer], x) 
        model = Model( pre_trained_model.input, x) 
        return model

    def __init__(self):
        self.twistDataLength = 10
        self.trainBatchSize = 70
        self.testBatchSize = 30
        
        print(tf.__version__)
        self.model = self._getInceptionModel()
        self.model.compile(
            optimizer='Adam',
            loss='mean_squared_error', 
            # metrics=[metrics.MeanSquaredError(name='mse'), metrics.MeanAbsoluteError(name='mae')])
            metrics=[metrics.MeanAbsoluteError(name='mae')])
            
        self.df = preprocessAllData('/home/majd/catkin_ws/src/basic_rl_agent/data/testing_data/allData.pkl')
        train_dataset = self.df.sample(frac=0.8, random_state=1)
        test_dataset = self.df.drop(labels=train_dataset.index, axis=0)
        test_dataset = test_dataset.sample(frac=1)
        # test_dataset = test_dataset.sample(frac=0.1, random_state=1)
        self.train_x = [train_dataset['images'].tolist(), train_dataset['vel'].tolist()]
        self.train_y = [train_dataset['positionControlPoints'].tolist(), train_dataset['yawControlPoints'].tolist()]
        self.test_x = [test_dataset['images'].tolist(), test_dataset['vel'].tolist()]
        self.test_y = [test_dataset['positionControlPoints'].tolist(), test_dataset['yawControlPoints'].tolist()]
        self.deleteInexistentImages()
    
    def _showRandomImages(self):
        for image in self.train_x[0:1]:
            img = cv2.imread(image)
            print(img.shape)
            cv2.imshow('image', img)
            cv2.waitKey(0)
    
    def train(self):
        training_generator = DataGenerator(self.train_x, self.train_y, self.trainBatchSize, twist_data_len=self.twistDataLength)
        testing_generator = DataGenerator(self.test_x, self.test_y, self.testBatchSize, twist_data_len=self.twistDataLength)

        # self.log_dir = "./logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # tensorboardCallback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1)

        modelSpcificationsDict = {'Model': 'inception top (not trained) for images, input layer with shape=(10,4) for twist data, 3 dense layers.', 'Model Input': 'a 320X240 RGB image, 40 twist data for linear vel on x, y, z and yaw angular velocity', 
            'Model Output': '12 floating points, The 3d Position Control Points', 
            'training Batch size': '{}'.format(self.trainBatchSize),
            'testing Batch size': '{}'.format(self.testBatchSize),
            'model transfer learning': 'the whole model is trained (even the CNN part)'
            }
        # tensorboardCallback = TensorBoardExtended(modelSpcificationsDict, log_dir=self.log_dir, histogram_freq=1)
        print('training...')
        try:
            history = self.model.fit(
                x=training_generator, epochs=10, 
                validation_data=testing_generator, validation_steps=5, 
                # callbacks=[tensorboardCallback],
                verbose=1, workers=4, use_multiprocessing=True)
        except KeyboardInterrupt:
            print('KeyboardInterrupt, model weights were saved.')
        finally:
            self.model.save_weights('./model_weights/weights{}.h5'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
        return history

    def test(self):
        self.model.load_weights('./model_weights/weights20210712-215937.h5')
        testBatchSize = 1
        gen = DataGenerator(self.train_x, self.train_y, testBatchSize, twist_data_len=self.twistDataLength)
        print('testing...')
        for i in range(10): #gen.__len__()):
            x, y = gen.__getitem__(i)
            self.model.fit(x, y, epochs=10)
            y_hat = self.model.predict(x)[0]
            positionCP_hat = y_hat[:12].reshape(4, 3).T
            positionCP_hat = np.concatenate([np.zeros((3, 1)), positionCP_hat], axis=1)
            # processing y:
            positionCP = (y[0])[:12]
            positionCP = np.concatenate([np.zeros((3,1)), positionCP.reshape(4, 3).T], axis=1)
            # plotting:
            acc = 100
            t_space = np.linspace(0, 1, acc)
            Phat_list = []
            P_list = []
            for ti in t_space:
                Phat = bezier4thOrder(positionCP_hat, ti) 
                Phat_list.append(Phat)
                P = bezier4thOrder(positionCP, ti)
                P_list.append(P)
            Phat_list = np.array(Phat_list)
            P_list = np.array(P_list)
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.set_aspect('equal')
            ax.plot3D(Phat_list[:, 0], Phat_list[:, 1], Phat_list[:, 2], 'r')
            ax.plot3D(P_list[:, 0], P_list[:, 1], P_list[:, 2], 'b')
            for p in positionCP_hat.T:
                ax.scatter(p[0], p[1], p[2], color='r')
            for p in positionCP.T:
                ax.scatter(p[0], p[1], p[2], color='b')
            plt.show()
        
                


    
    def deleteInexistentImages(self):
        for i, img in enumerate(self.train_x[0]):
            if os.path.isfile(img) == False:
                self.train_x[0].pop(i)
                self.train_x[1].pop(i)
                self.train_y[0].pop(i)
                self.train_y[1].pop(i)
        for i, img in enumerate(self.test_x[0]):
            if os.path.isfile(img) == False:
                self.test_x[0].pop(i)
                self.test_x[1].pop(i)
                self.test_y[0].pop(i)
                self.test_y[1].pop(i)
        
def dataGeneratorTester():
    df = preprocessAllData('/home/majd/catkin_ws/src/basic_rl_agent/data/testing_data/allData.pkl')
    train_dataset = df.sample(frac=0.8, random_state=1)
    test_dataset = df.drop(labels=train_dataset.index, axis=0)
    # test_dataset = test_dataset.sample(frac=0.5, random_state=1)
    train_x = [train_dataset['images'].tolist(), train_dataset['vel'].tolist()]
    train_y = [train_dataset['positionControlPoints'].tolist(), train_dataset['yawControlPoints'].tolist()]
    gen = DataGenerator(train_x, train_y, batch_size=100, twist_data_len=10) 
    for k in range(20):
        for i in range(gen.__len__()):
            x, y = gen.__getitem__(i)
            print(k, i, x[0].shape, x[1].shape, y.shape)
            assert x[0].shape[0] == x[1].shape[0] and y.shape[0] == x[1].shape[0], 'assertion failed'

def dataFrameTester():
    df = preprocessAllData('/home/majd/catkin_ws/src/basic_rl_agent/data/testing_data/allData.pkl')
    print(df['yawControlPoints'])
    
def main():
    trainer = Trainer() 
    # trainer.train()
    trainer.test()


if __name__=='__main__':
    main()