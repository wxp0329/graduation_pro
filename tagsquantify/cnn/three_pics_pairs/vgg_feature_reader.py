# coding:utf-8
import tensorflow as tf
import numpy as np
import os
from PIL import Image

from tagsquantify.cnn.three_pics_pairs import Three_net_enforce

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('pair_file',
                           r'F:\NUS_dataset\graduate_data\219388_2003_indexes_0.2_three_pair.dat',
                           """three pairs files dir.""")
tf.app.flags.DEFINE_string('vgg_feature_file',
                           r'F:\NUS_dataset\graduate_data\vgg16_fc8_219388_2003_pics_mat.npy',
                           """three pairs files dir.""")


class InputUtil:
    def __init__(self):
        # read pair file
        with open(FLAGS.pair_file) as fr:
            self.paths = fr.readlines()
        # load vgg features
        self.pic_mat = np.load(FLAGS.vgg_feature_file)
        self.pic_mat=(self.pic_mat-np.mean(self.pic_mat))/np.std(self.pic_mat)
        print('219388_2003 vgg_feature loaded over!！')
        self.start = 0
        self.end = 0

    def next_batch(self, batch_size=Three_net_enforce.FLAGS.batch_size / 3,
                   shuffle_=True):  # due to concatenate([i_s, j_s, k_s]),so dived by 3
        if self.start == 0 and shuffle_:
            np.random.shuffle(self.paths)
        self.end = self.start + batch_size
        if self.end > len(self.paths):
            self.end = len(self.paths)
            self.start = self.end - batch_size

        i_s = []
        j_s = []
        k_s = []
        for line in self.paths[int(self.start):int(self.end)]:
            line = line.strip().split(' ')
            i_s.append(self.pic_mat[int(line[0])])
            j_s.append(self.pic_mat[int(line[1])])
            k_s.append(self.pic_mat[int(line[2])])

        if self.end == len(self.paths):
            self.start = 0
        else:
            self.start = self.end
            # 一维数组
        return np.concatenate([i_s, j_s, k_s])
