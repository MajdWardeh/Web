import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as k
from tensorflow.keras.applications.inception_v3 import InceptionV3


print(tf.__version__)
# local_weights_file = 'inception_weights/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

# pre_trained_model = InceptionV3(input_shape = (98, 98, 3), 
#                                 include_top = False, 
#                                 weights = None)
# pre_trained_model.load_weights(local_weights_file)

# for layer in pre_trained_model.layers:
#   layer.trainable = False
  
# pre_trained_model.summary()

# last_layer = pre_trained_model.get_layer('mixed7')
# print('last layer output shape: ', last_layer.output_shape)
# last_output = last_layer.output