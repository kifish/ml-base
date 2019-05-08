#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =======================================
# File Name : model.py
# Purpose : build a compute graph
# Creation Date : 2019-02-19 18:57
# Last Modified :
# Created By : sunpeiqin
# =======================================

import tensorflow as tf
import tensorflow.contrib as tf_contrib
from common import config


class Model():
    def __init__(self):
        # set the initializer of conv_weight and conv_bias
        self.weight_init = tf_contrib.layers.variance_scaling_initializer(factor=1.0,
                                mode='FAN_IN', uniform=False)
        self.bias_init = tf.zeros_initializer()
        self.reg = tf_contrib.layers.l2_regularizer(config.weight_decay)

    def gaussian_kernel_init(self,shape, mean=0., std=1.0):
        edge_len = shape[0]
        d = tf.distributions.Normal(mean, std)
        if edge_len % 2 == 1:
            vals = d.prob(tf.range(start=-(edge_len // 2), limit=(edge_len // 2) + 1, delta=1, dtype=tf.float32))
        else:
            vals = d.prob(tf.range(start=-(edge_len - 1), limit=(edge_len - 1) + 1, delta=2, dtype=tf.float32))
        gauss_kernel = tf.einsum('i,j->ij', vals, vals)
        gauss_kernel = gauss_kernel / tf.reduce_sum(gauss_kernel)
        gauss_kernel = tf.stack([gauss_kernel] * shape[2])
        gauss_kernel = tf.stack([gauss_kernel] * shape[3])
        gauss_kernel = tf.transpose(gauss_kernel, perm=[3, 2, 0, 1])
        return gauss_kernel

    def _conv_layer(self, name, inp, kernel_shape, stride, padding='SAME',is_training=False):
        with tf.variable_scope(name) as scope:
            conv_filter = tf.get_variable(name='filter', shape=kernel_shape,
                                          initializer=self.weight_init, regularizer=self.reg)
            conv_bias = tf.get_variable(name='bias', shape=kernel_shape[-1],
                                        initializer=self.bias_init)
            x = tf.nn.conv2d(inp, conv_filter, strides=[1, stride, stride, 1],
                             padding=padding, data_format='NHWC')
            x = tf.nn.bias_add(x, conv_bias, data_format='NHWC')
            x = tf.layers.batch_normalization(x, axis=3, training=is_training)
            x = tf.nn.relu(x)
        return x

    def _pool_layer(self, name, inp, ksize, stride, shape = None,padding='SAME', mode='MAX'):
        assert (mode in ['MAX', 'AVG'] or isinstance(mode,int)), 'the mode of pool must be MAX or AVG or Interger'
        if mode == 'MAX':
            x = tf.nn.max_pool(inp, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1],
                               padding=padding, name=name, data_format='NHWC')
        elif mode == 'AVG':
            x = tf.nn.avg_pool(inp, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1],
                               padding=padding, name=name, data_format='NHWC')
        else:
            if mode < 10:
                with tf.variable_scope(name) as scope:
                    gauss_kernel = tf.get_variable(name='gauss_kernel',initializer=self.gaussian_kernel_init(shape),trainable=False)
                    p = mode
                    x = tf.pow(inp,p)
                    x = tf.nn.conv2d(x,gauss_kernel,strides=[1,stride,stride,1],padding=padding,name = name,data_format='NHWC')
                    x = tf.pow(x,1/p)
            else:
                with tf.variable_scope(name) as scope:
                    x = tf.nn.max_pool(inp,ksize=[1,shape[0],shape[0],1],strides=[1,stride,stride,1],padding=padding,name = name,data_format='NHWC')
        return x

    def _fc_layer(self, name, inp, units, dropout=0.5):
        with tf.variable_scope(name) as scope:
            shape = inp.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(inp, [-1, dim]) # flatten
            if dropout > 0:
                x = tf.nn.dropout(x, keep_prob=dropout, name='dropout')
            x = tf.layers.dense(x, units, kernel_initializer=self.weight_init,
                                bias_initializer=self.bias_init, kernel_regularizer=self.reg)
        return x

    #def _softmax_layer(self, name, inp):
    #    x = tf.nn.softmax(inp, name=name)
    #    return x

    def build(self):
        data = tf.placeholder(tf.float32, shape=(None,)+config.image_shape+(config.nr_channel,),
                              name='data')
        label = tf.placeholder(tf.int32, shape=(None,), name='label')
        # convert the format of label to one-hot
        label_onehot = tf.one_hot(label, config.nr_class, dtype=tf.int32)
        # a setting for bn
        is_training = tf.placeholder(tf.bool, name='is_training')

        # conv1
        x = self._conv_layer(name='conv1', inp=data,
                             kernel_shape=[3, 3, config.nr_channel, 16], stride=1,
                             is_training=is_training) # Nx32x32x16
        x = self._pool_layer(name='pool1', inp=x, ksize=None, padding='SAME',shape= [3,3,16,16],stride=2, mode=100) # Nx16x16x16

        # conv2
        x = self._conv_layer(name='conv21', inp=x, kernel_shape=[3, 3, 16, 32],
                             stride=1, is_training=is_training)
        x = self._conv_layer(name='conv22', inp=x, kernel_shape=[3, 3, 32, 32],
                             stride=1, is_training=is_training) # Nx16x16x32
        x = self._pool_layer(name='pool2', inp=x, ksize=None, padding='SAME',shape= [4,4,32,32],stride=2, mode=100) # Nx8x8x32

        # conv3
        x = self._conv_layer(name='conv31', inp=x, kernel_shape=[3, 3, 32, 64],
                             stride=1, is_training=is_training)
        x = self._conv_layer(name='conv32', inp=x, kernel_shape=[3, 3, 64, 64],
                             stride=1, is_training=is_training) # Nx8x8x64
        x = self._pool_layer(name='pool3', inp=x, ksize=None, padding = 'SAME',shape= [5,5,64,64],stride=2, mode=100) # Nx4x4x64

        # conv4
        x = self._conv_layer(name='conv41', inp=x, kernel_shape=[3, 3, 64, 128],
                             stride=1, is_training=is_training)
        x = self._conv_layer(name='conv42', inp=x, kernel_shape=[3, 3, 128, 128],
                             stride=1, is_training=is_training) # Nx4x4x128
        x = self._pool_layer(name='pool4', inp=x, ksize=None, padding='VALID', shape= [4, 4, 128, 128], stride=1, mode=100) # Nx1x1x128

        # fc1
        logits = self._fc_layer(name='fc1', inp=x, units=config.nr_class, dropout=0)

        placeholders = {
            'data': data,
            'label': label,
            'is_training': is_training,
        }
        return placeholders, label_onehot, logits