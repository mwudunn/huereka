import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import ColorOps

class ColorSuggestModel:
    def __init__(self, config):
        self.model_params = config['model_params']
        self.training_params = config['training_params']

    def buildModel(self, image):
        x = tf.cast(image, tf.float32)
        with tf.variable_scope("simpleCnnModel"):
            for channels in self.model_params['channels']:
                x = tf.compat.v1.layers.conv2d(x, channels, self.model_params['size'], self.model_params['stride'],
                                     padding='SAME', activation=tf.nn.relu)
                x = tf.compat.v1.layers.max_pooling2d(x, self.model_params['pool_size'], self.model_params['pool_strides'])
            x = tf.reshape(x, (-1,2*2*self.model_params['channels'][-1]))
            for dense_size in self.model_params['dense']:
                x = tf.layers.dense(x, dense_size, activation=tf.nn.relu)
            x = tf.layers.dense(x, self.model_params['num_colors'] * 3)
        out = tf.reshape(x, (-1,self.model_params['num_colors'],3))

        # out = tf.clip_by_value(out, 0., 1.)
        out = tf.sigmoid(out)

        return out

    def buildOpt(self, model, labels):
        colors = ColorOps.sRGB_to_XYZ(model)
        colors = ColorOps.XYZ_to_LAB(colors)
        labels = ColorOps.sRGB_to_XYZ(labels)
        labels = ColorOps.XYZ_to_LAB(labels)
        diff = ColorOps.deltaE_2000(colors, labels)
        loss = tf.reduce_mean(tf.square(diff))

        lr = tf.Variable(1e-3,trainable=False,name='lr')
        lrPH = tf.placeholder(tf.float32,(),name='lrPH')
        lr_assign = tf.assign(lr,lrPH)
        global_step = tf.compat.v1.train.get_or_create_global_step()

        decay = float(self.training_params['weight_decay'])
        optimizer = tf.contrib.opt.AdamWOptimizer(decay, lr)
        opt = optimizer.minimize(loss, global_step=global_step)

        return dict(opt=opt, lr_assign=lr_assign, lrPH=lrPH, loss=loss, global_step=global_step)