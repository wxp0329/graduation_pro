import inspect
import os

import numpy as np
import tensorflow as tf
import time

VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg16:
    def __init__(self, vgg16_npy_path=None, train_mode=False):
        # if vgg16_npy_path is None:
        #     path = inspect.getfile(vgg16_npy_path)
        #     path = os.path.abspath(os.path.join(path, os.pardir))
        #     path = os.path.join(path, "vgg16.npy")
        #     vgg16_npy_path = path
        self.var_dict = {}
        self.data_dict = {}
        self.train_mode = train_mode
        if vgg16_npy_path != None:
            self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        print("npy file loaded")

    def inference(self, rgb):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
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

        # self.conv1_1 = self.conv_layer(rgb, 3, 64, "conv1_1", trainable=self.train_mode)
        # self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64, "conv1_2", trainable=self.train_mode)
        # self.pool1 = self.max_pool(self.conv1_2, 'pool1')
        #
        # self.conv2_1 = self.conv_layer(self.pool1, 64, 128, "conv2_1", trainable=self.train_mode)
        # self.conv2_2 = self.conv_layer(self.conv2_1, 128, 128, "conv2_2", trainable=self.train_mode)
        # self.pool2 = self.max_pool(self.conv2_2, 'pool2')
        #
        # self.conv3_1 = self.conv_layer(self.pool2, 128, 256, "conv3_1", trainable=self.train_mode)
        # self.conv3_2 = self.conv_layer(self.conv3_1, 256, 256, "conv3_2", trainable=self.train_mode)
        # self.conv3_3 = self.conv_layer(self.conv3_2, 256, 256, "conv3_3", trainable=self.train_mode)
        # self.pool3 = self.max_pool(self.conv3_3, 'pool3')
        #
        # self.conv4_1 = self.conv_layer(self.pool3, 256, 512, "conv4_1", trainable=self.train_mode)
        # self.conv4_2 = self.conv_layer(self.conv4_1, 512, 512, "conv4_2", trainable=self.train_mode)
        # self.conv4_3 = self.conv_layer(self.conv4_2, 512, 512, "conv4_3", trainable=self.train_mode)
        # self.pool4 = self.max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer( rgb , 512, 512, "conv5_1", trainable=self.train_mode)
        self.conv5_2 = self.conv_layer(self.conv5_1, 512, 512, "conv5_2", trainable=self.train_mode)
        self.conv5_3 = self.conv_layer(self.conv5_2, 512, 512, "conv5_3", trainable=self.train_mode)
        self.pool5 = self.max_pool(self.conv5_3, 'pool5')

        shape = int(np.prod(self.pool5.get_shape()[1:]))
        self.fc6 = self.fc_layer(self.pool5, shape, 4096, "fc6", trainable=self.train_mode)
        assert self.fc6.get_shape().as_list()[1:] == [4096]
        self.relu6 = tf.nn.relu(self.fc6)
        if self.train_mode:
            self.relu6 = tf.nn.dropout(self.relu6, 0.5)

        self.fc7 = self.fc_layer(self.relu6, 4096, 4096, "fc7", trainable=self.train_mode)
        self.relu7 = tf.nn.relu(self.fc7)
        if self.train_mode:
            self.relu7 = tf.nn.dropout(self.relu7, 0.5)
        #
        # self.fc8 = self.fc_layer(self.relu7, 4096, 1000, "fc8", trainable=True)
        #
        # self.prob = tf.nn.softmax(self.fc8, name="prob")

        return self.relu7

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    # def conv_layer(self, bottom, name):
    #     with tf.variable_scope(name):
    #         filt = self.get_conv_filter(name)
    #
    #         conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
    #
    #         conv_biases = self.get_bias(name)
    #         bias = tf.nn.bias_add(conv, conv_biases)
    #
    #         relu = tf.nn.relu(bias)
    #         return relu
    #
    # def fc_layer(self, bottom, name):
    #     with tf.variable_scope(name):
    #         shape = bottom.get_shape().as_list()
    #         dim = 1
    #         for d in shape[1:]:
    #             dim *= d
    #         x = tf.reshape(bottom, [-1, dim])
    #
    #         weights = self.get_fc_weight(name)
    #         biases = self.get_bias(name)
    #
    #         # Fully connected layer. Note that the '+' operation automatically
    #         # broadcasts the biases.
    #         fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
    #
    #         return fc
    #
    # def get_conv_filter(self, name):
    #     return tf.constant(self.data_dict[name][0], name="filter")
    #
    # def get_bias(self, name):
    #     return tf.constant(self.data_dict[name][1], name="biases")
    #
    # def get_fc_weight(self, name):
    #     return tf.constant(self.data_dict[name][0], name="weights")

    def conv_layer(self, bottom, in_channels, out_channels, name, trainable):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name, trainable)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)
            weight_decay = tf.multiply(tf.nn.l2_loss(relu), 0.0005, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
            return relu

    def fc_layer(self, bottom, in_size, out_size, name, trainable):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name, trainable)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
            weight_decay = tf.multiply(tf.nn.l2_loss(fc), 0.004, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
            return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name, trainable):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var(initial_value, name, 0, name + "_filters", trainable)

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases", trainable)

        return filters, biases

    def get_fc_var(self, in_size, out_size, name, trainable):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var(initial_value, name, 0, name + "_weights", trainable)

        initial_value = tf.truncated_normal([out_size], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases", trainable)

        return weights, biases

    def get_var(self, initial_value, name, idx, var_name, trainable):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value
        var=tf.Variable(value, name=var_name, trainable=trainable)
        self.var_dict[(name, idx)] = var

        # print var_name, var.get_shape().as_list()
        # print('var: {} ,and initial_value: {} shape'.format(var.get_shape(),initial_value.get_shape()))
        assert var.get_shape() == initial_value.get_shape()

        return var

    def save_npy(self, sess, npy_path="./vgg16-save.npy"):
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
