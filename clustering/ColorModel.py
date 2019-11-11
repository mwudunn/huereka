import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class ColorSuggestModel:
    def __init__(self, config):
        self.model_params = config['model_params']

    def buildModel(image):
        x = tf.cast(image, tf.float32)
        with tf.variable_scope("cnn"):
            for channels in self.model_params['channels']:
                x = tf.compat.v1.layers.conv2d(x, channels, self.model_params['size'], self.model_params['stride'],
                                     padding='SAME', activation=tf.nn.relu)
                x = tf.compat.v1.layers.max_pooling2d(x, self.model_params['pool_size'], self.model_params['pool_strides'])
            x = tf.reshape(x, (-1,2*2*self.model_params['channels'][-1]))
        for dense_size in self.model_params['dense']:
            x = tf.layers.dense(x, dense_size, activation=tf.nn.relu)
        x = tf.layers.dense(x, self.model_params['num_colors'] * 3)
        out = tf.reshape(x, (-1,self.model_params['num_colors'],3))

        return out

    def buildOpt():
        pass