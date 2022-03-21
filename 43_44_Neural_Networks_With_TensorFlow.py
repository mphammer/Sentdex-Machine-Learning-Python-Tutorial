'''
https://www.tensorflow.org/

Tensor Flow: 
- TensorFlow is an end-to-end open source platform for machine learning. It has a comprehensive, flexible ecosystem of tools, libraries.
- The core open source library to help you develop and train ML models.

https://www.tensorflow.org/install

Install TensorFlow with Python's pip package manager.
$ pip install tensorflow

A Tensor is an array-like object. 
TensorFlow basically just applies functions to arrays. 
'''

import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

with tf.compat.v1.Session() as sess:
    # Building the Dataflow graph: 
    # tf.constant() - Creates a constant tensor from a tensor-like object.
    # https://www.tensorflow.org/api_docs/python/tf/constant
    x1 = tf.constant(5)
    x2 = tf.constant(6)
    result_tensor = tf.multiply(x1, x2)
    print(result_tensor)
    
    # Execute the graph and store the result in 'output'
    output = sess.run(result_tensor)
    print(output)