import os
import tensorflow as tf

import numpy as np
import time
import inspect, re

VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg19:
    """
    A trainable version VGG19.
    """

    def activation_summary(self, x):
        """Helper to create summaries for activations.

        Creates a summary that provides a histogram of activations.
        Creates a summary that measures the sparsity of activations.

        Args:
          x: Tensor
        Returns:
          nothing
        """
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
        TOWER_NAME = 'tower'
        tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
        tf.summary.histogram(tensor_name + '/activations', x)
        tf.summary.scalar(tensor_name + '/sparsity',
                          tf.nn.zero_fraction(x))

    def __init__(self, vgg19_npy_path=None, train_mode=False):
        if vgg19_npy_path is not None:
            self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
        else:
            self.data_dict = None
        print(self.data_dict.keys())
        self.train_mode = train_mode
        self.var_dict = {}

    def inference(self, bgr):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        :param train_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
        """

        # rgb_scaled = rgb * 255.0
        #
        # # Convert RGB to BGR
        # red, green, blue = tf.split(3, 3, rgb_scaled)
        # assert red.get_shape().as_list()[1:] == [224, 224, 1]
        # assert green.get_shape().as_list()[1:] == [224, 224, 1]
        # assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        # bgr = tf.concat(3, [
        #     blue - VGG_MEAN[0],
        #     green - VGG_MEAN[1],
        #     red - VGG_MEAN[2],
        # ])
        # assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        self.conv1_1 = self.conv_layer(bgr, 3, 64, "conv1_1", trainable=False)
        self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64, "conv1_2", trainable=False)
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, 64, 128, "conv2_1", trainable=False)
        self.conv2_2 = self.conv_layer(self.conv2_1, 128, 128, "conv2_2", trainable=False)
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, 128, 256, "conv3_1", trainable=False)
        self.conv3_2 = self.conv_layer(self.conv3_1, 256, 256, "conv3_2", trainable=False)
        self.conv3_3 = self.conv_layer(self.conv3_2, 256, 256, "conv3_3", trainable=False)
        self.conv3_4 = self.conv_layer(self.conv3_3, 256, 256, "conv3_4", trainable=False)
        self.pool3 = self.max_pool(self.conv3_4, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, 256, 512, "conv4_1", trainable=False)
        self.conv4_2 = self.conv_layer(self.conv4_1, 512, 512, "conv4_2", trainable=False)
        self.conv4_3 = self.conv_layer(self.conv4_2, 512, 512, "conv4_3", trainable=False)
        self.conv4_4 = self.conv_layer(self.conv4_3, 512, 512, "conv4_4", trainable=False)
        self.pool4 = self.max_pool(self.conv4_4, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, 512, 512, "conv5_1", trainable=False)
        self.conv5_2 = self.conv_layer(self.conv5_1, 512, 512, "conv5_2", trainable=False)
        self.conv5_3 = self.conv_layer(self.conv5_2, 512, 512, "conv5_3", trainable=False)
        self.conv5_4 = self.conv_layer(self.conv5_3, 512, 512, "conv5_4", trainable=False)
        self.pool5 = self.max_pool(self.conv5_4, 'pool5')

        self.fc6 = self.fc_layer(self.pool5, 25088, 4096, "fc6",
                                 trainable=False)  # 25088 = ((224 / (2 ** 5)) ** 2) * 512
        self.relu6 = tf.nn.relu(self.fc6)
        self.activation_summary(self.relu6)
        if self.train_mode:
            self.relu6 = tf.nn.dropout(self.relu6, 0.5)

        self.fc7 = self.fc_layer(self.relu6, 4096, 4096, "fc7", trainable=False)
        self.relu7 = tf.nn.relu(self.fc7)
        # self.activation_summary(self.relu7)
        # if self.train_mode:
        #     self.relu7 = tf.nn.dropout(self.relu7, 0.5)
        #
        # self.fc8 = self.fc_layer(self.relu7, 4096, 1000, "fc8", trainable=False)
        # self.activation_summary(self.fc8)
        # self.prob = tf.nn.softmax(self.fc8, name="prob")

        return self.relu7

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, in_channels, out_channels, name, trainable):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name, trainable=trainable)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)
            self.activation_summary(relu)
            return relu

    def fc_layer(self, bottom, in_size, out_size, name, trainable):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name, trainable=trainable)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name, trainable):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var(initial_value, name, 0, name + "_filters", trainable=trainable)

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases", trainable=trainable)

        return filters, biases

    def get_fc_var(self, in_size, out_size, name, trainable):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var(initial_value, name, 0, name + "_weights", trainable=trainable)

        initial_value = tf.truncated_normal([out_size], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases", trainable=trainable)

        return weights, biases

    def get_var(self, initial_value, name, idx, var_name, trainable):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        var = tf.Variable(value, name=var_name, trainable=trainable)

        self.var_dict[(name, idx)] = var

        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()

        return var

    def save_npy(self, sess, npy_path="./vgg19-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in self.var_dict.items():
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print("file saved", npy_path)
        return npy_path
