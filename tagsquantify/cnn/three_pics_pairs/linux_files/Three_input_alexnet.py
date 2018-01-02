# coding:utf-8
import tensorflow as tf
import numpy as np
import os
from PIL import Image

from tagsquantify.cnn.three_pics_pairs.linux_files import Three_net_alexnet

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('img_size', 224,
                            """Number of images to process in a batch.""")

tf.app.flags.DEFINE_string('pair_dir', r'F:\NUS_dataset\graduate_data\169345_0.2_filename_three_pair.txt',
                           """three pairs files dir.""")
tf.app.flags.DEFINE_string('imgs_dir',
                           r'F:\NUS_dataset\images_220841',
                           """Path to the NUS data directory.""")


class InputUtil:
    def __init__(self):

        with open(FLAGS.pair_dir) as fr:  # 文件格式：i j k 代表pic名字索引
            self.paths = fr.readlines()
        self.start = 0
        self.end = 0
        print('read pair_files over !!!')

    # 获取该图片对应的输入向量
    def getimg(self, str1):
        im = Image.open(str1)
        re_img = np.array(im.resize((FLAGS.img_size, FLAGS.img_size)), dtype=np.float32)
        # Subtract off the mean and divide by the variance of the pixels.
        return (re_img - np.mean(re_img)) / np.std(re_img)

    def next_batch(self, batch_size=Three_net_alexnet.FLAGS.batch_size, shuffle_=True):
        if self.start == 0 and shuffle_:
            np.random.shuffle(self.paths)
        self.end = self.start + batch_size / 3
        if self.end > len(self.paths):
            self.end = len(self.paths)
            self.start = self.end - batch_size / 3

        i_s = []
        j_s = []
        k_s = []
        for line in self.paths[int(self.start):int(self.end)]:
            line = line.strip().split(' ')
            i_s.append(self.getimg(os.path.join(FLAGS.imgs_dir, line[0] + '.jpg')))
            j_s.append(self.getimg(os.path.join(FLAGS.imgs_dir, line[1] + '.jpg')))
            k_s.append(self.getimg(os.path.join(FLAGS.imgs_dir, line[2] + '.jpg')))
        if self.end == len(self.paths):
            self.start = 0
        else:
            self.start = self.end
        # 一维数组
        return np.concatenate([i_s, j_s, k_s])


if __name__ == '__main__':
    a = InputUtil()
    a.next_batch()
