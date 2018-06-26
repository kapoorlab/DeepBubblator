'''
Created on Jun 6, 2018

@author: aimachine
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#Imports

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

if __name__ == "__main__":
      tf.app.run()
    
    #Mode is for running it in training, evaluation or prediction mode
def cnn_model_fn(features, labels, mode):   
     
     """ Model function for CNN """
     #Input layer
     input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
     
     # Convolutional Layer #1
     
     convLayerA = tf.layers.conv2d(inputs = input_layer, filters = 32, kernel_size= [5,5], padding = "same", activation = tf.nn.relu)
     poolLayerA = tf.layers.max_pooling2d(inputs = convLayerA, pool_size=[2, 2], strides = 2)
     
     #Convolutional layer 2
     
     convLayerB = tf.layers.conv2d(inputs = poolLayerA, filters = 64, kernel_size = [5, 5], padding = "same", activation = tf.nn.relu)
     poolLayerB = tf.layers.max_pooling2d(inputs = convLayerB, pool_size = [2, 2], strides = 2)
     
     