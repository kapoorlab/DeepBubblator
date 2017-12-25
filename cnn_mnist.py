'''
Created on 25 Dec 2017

@author: varunkapoor
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
from keras.layers.core import Activation

tf.logging.set_verbosity(tf.logging.INFO)

N_channels = tf.placeholder(tf.int32)  
BatchSize = tf.placeholder(tf.int32)
ImagesizeX = tf.placeholder(tf.int32)
ImagesizeY = tf.placeholder(tf.int32)
"Initialize variables"
init = tf.global_variables_initializer()
"Create a session"
sess = tf.Session()
"Initialize variables"
sess.run(init)

"Single channel image"
N_channels = 1

"Dynamically compute the batch size"
BatchSize = -1

"Image dimension along X"
ImagesizeX = 28

"Image dimension along Y"
ImagesizeY = 28


print("Channels: %s BatchSize %s" % (N_channels, BatchSize))


def cnn_model_fn(features, labels, mode):
 """ Model function for CNN."""
 #Input Layer
 
 input_layer = tf.reshape(features["x"], [BatchSize, ImagesizeX, ImagesizeY, N_channels])
 
 #Create a convolutional layer (outputshape = -1, 28,28,32)
 
 conv1 = tf.layers.conv2d(inputs = input_layer, filters = 32, kernel_size= [5, 5], padding="same", activation= tf.nn.relu)
 
 #Create a pooling layer (outputshape = -1, 14, 14, 32)
 
 pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size = [2, 2], strides = 2)
 
 #Create a second convolutional layer (outputshape = -1, 14,14,64)

 conv2 = tf.layers.conv2d(inputs = pool1, filters = 64, kernel_size= [5, 5], padding = "same", activation = tf.nn.relu)
 
   #Create a pooling layer (outputshape = -1, 7, 7, 64)
 
 pool2 = tf.layers.max_pooling2d(inputs = conv2, pool_size = [2, 2], strides = 2)
 
 
 #Build a dense layer by making a flat pooling layer
 
 pool2_flat = tf.reshape(pool2, 7*7*64)
 
 dense = tf.layers.dense(inputs = pool2_flat, units= 1024, activation=tf.nn.relu)
 
 droupout = tf.layers.dropout(inputs= dense, rate = 0.4, training = mode == tf.estimator.ModeKeys.TRAIN)
 
 #Final Layer logits has shape of batchsize, 10
 
 logits = tf.layers.dense(inputs = droupout, units = 10)
 
 
 predictions = {"classes" : tf.argmax(input=logits, axis = 1), "probabilities": tf.nn.softmax(logits, name = "softmax_tensor")}
 
 if mode == tf.estimator.ModeKeys.PREDICT:
     return tf.estimator.EstimatorSpec(mode = mode, predictions = predictions)
 
 
 
 
# Application logic

if __name__ == "__main__":
     tf.app.run()
  
    
    
 