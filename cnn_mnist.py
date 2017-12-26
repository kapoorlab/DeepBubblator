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
 
 pool2_flat = tf.reshape(pool2, [BatchSize, 7*7*64])
 
 dense = tf.layers.dense(inputs = pool2_flat, units= 1024, activation=tf.nn.relu)
 
 droupout = tf.layers.dropout(inputs= dense, rate = 0.4, training = mode == tf.estimator.ModeKeys.TRAIN)
 
 #Final Layer logits has shape of batchsize, 10
 
 logits = tf.layers.dense(inputs = droupout, units = 10)
 
 
 predictions = {"classes" : tf.argmax(input=logits, axis = 1), "probabilities": tf.nn.softmax(logits, name = "softmax_tensor")}
 
 if mode == tf.estimator.ModeKeys.PREDICT:
     return tf.estimator.EstimatorSpec(mode = mode, predictions = predictions)
 
 
 "Loss function"
 
 onehot_labels = tf.one_hot(indices= tf.cast(labels, tf.int32), depth = 10)
 loss = tf.losses.softmax_cross_entropy(onehot_labels = onehot_labels, logits = logits)
 
 "Here comes the SGD"
 if mode == tf.estimator.ModeKeys.TRAIN:
     optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
     train_op = optimizer.minimize(loss = loss, global_step = tf.train.get_global_step())
     return tf.estimator.EstimatorSpec(mode = mode, loss = loss, train_op = train_op)
 
 
 if mode == tf.estimator.ModeKeys.EVAL:
 # Add evaluation metrics (for EVAL mode)
     eval_metric_ops = {
         "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
     return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
     
     mnist = tf.contrib.learn.datasets.load_dataset("mnist")
     train_data = mnist.train.images
     train_labels = np.asarray(mnist.train.labels, dtype = np.int32)
     eval_data = mnist.test.images
     eval_labels = np.asarray(mnist.test.labels, dtype = np.int32)
     mnist_classifier = tf.estimator.Estimator(model_fn = cnn_model_fn, model_dir="/Users/varunkapoor/mnist_convnet_model")     
     tensors_to_log = {"probabilities": "softmax_tensor"}
     logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)
     #Train the model
     train_input_fn = tf.estimator.inputs.numpy_input_fn(
     x={"x": train_data},
     y=train_labels,
     batch_size=100,
     num_epochs=None,
     shuffle=True)
     mnist_classifier.train(
     input_fn=train_input_fn,
     steps=1,
     hooks=[logging_hook])
 # Evaluate the model and print results
     eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
     eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
     print(eval_results) 
 

if __name__ == "__main__":
     print(__name__)

     tf.app.run()
  
    
    
 