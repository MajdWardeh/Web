import sys

from tensorflow.python.keras import models

sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import datetime
import numpy as np
import math
import cv2
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, backend as k
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.metrics as metrics
from tensorflow.keras.utils import Sequence

# from tensorflow.python.keras.callbacks import TensorBoard
from keras.callbacks import TensorBoard

# from data_preprocessing import preprocessAllData
# from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3

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

    # print(df)
    return df

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

class DataGenerator(Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, index):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # x_batch = np.zeros((self.batch_size, 100, 200, 3), dtype='float32')
        # y_batch = np.zeros((self.batch_size, 12), dtype='float32')
        x_batch = []
        y_batch = []

        # fill up the batch
        for row in range(min(self.batch_size, len(self.x)-index*self.batch_size)):
            image = cv2.imread(self.x[index*self.batch_size + row])
            if image is None:
                continue
            image = cv2.resize(image, (240, 320))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32)
          
            x_batch.append(image)
            y_batch.append(np.array(self.y[index*self.batch_size + row][:]).reshape((12, )) )

        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)
        # Normalize inputs
        x_batch = x_batch/255.
        return (x_batch, y_batch)

class Trainer:

    def _getInceptionModel(self):
        local_weights_file = '../inception_weights/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
        pre_trained_model = InceptionV3(input_shape = (240, 320, 3), 
                                        include_top = False, 
                                        weights = None)
        pre_trained_model.load_weights(local_weights_file)

        for layer in pre_trained_model.layers:
            layer.trainable = False

        last_layer = pre_trained_model.get_layer('mixed7')
        # print('last layer output shape: ', last_layer.output_shape)
        last_output = last_layer.output

        x = layers.Flatten()(last_output)
        # Add a fully connected layer with 1,024 hidden units and ReLU activation
        x = layers.Dense(1024, activation='relu')(x)
        # Add a dropout rate of 0.2
        x = layers.Dropout(0.2)(x)  
        x = layers.Dense(512, activation='relu')(x)                
        # output layer:
        x = layers.Dense(12, activation=None)(x) 
        model = Model( pre_trained_model.input, x) 
        return model

    def __init__(self):
        
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
        # test_dataset = test_dataset.sample(frac=0.5, random_state=1)
        self.train_x = train_dataset['images'].tolist()
        self.train_y = train_dataset['positionControlPoints'].tolist()
        self.test_x = test_dataset['images'].tolist()
        self.test_y = test_dataset['positionControlPoints'].tolist()
        self.deleteInexistentImages()

        self.log_dir = "./logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    

    def temp(self):
        print(self.df.head())

    def _showRandomImages(self):
        for image in self.train_x[0:1]:
            img = cv2.imread(image)
            print(img.shape)
            cv2.imshow('image', img)
            cv2.waitKey(0)
    
    def train(self):
        training_generator = DataGenerator(self.train_x, self.train_y, 100)
        testing_generator = DataGenerator(self.test_x, self.test_y, 100)
        # tensorboardCallback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1)
        modelSpcificationsDict = {'Model': 'inception top (not trained), 3 dense layers', 'Model Input': 'a 320X240 RGB image', 
            'Model Output': '12 floating points, The 3d Position Control Points'}
        tensorboardCallback = TensorBoardExtended(modelSpcificationsDict, log_dir=self.log_dir, histogram_freq=1)
        print(modelSpcificationsDict)
        print('training...')
        history = self.model.fit(
            x=training_generator, epochs=30, 
            validation_data=testing_generator, validation_steps=5, 
            callbacks=[tensorboardCallback],
            verbose=1, workers=4, use_multiprocessing=True)
        return history
    
    def deleteInexistentImages(self):
        for i, img in enumerate(self.train_x):
            if os.path.isfile(img) == False:
                self.train_x.pop(i)
                self.train_y.pop(i)
        for i, img in enumerate(self.test_x):
            if os.path.isfile(img) == False:
                self.test_x.pop(i)
                self.test_y.pop(i)
        


    

def main():
    trainer = Trainer() 
    trainer.train()
    # trainer.temp()



if __name__=='__main__':
    main()