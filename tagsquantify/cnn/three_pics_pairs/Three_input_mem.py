# coding:utf-8
import tensorflow as tf
import numpy as np
import os
from PIL import Image

FLAGS = tf.app.flags.FLAGS



tf.app.flags.DEFINE_string('pair_dir', '/home/wangxiaopeng/NUS_dataset/218838_com_three_pair.txt',
                           """three pairs files dir.""")


class InputUtil:
    def __init__(self):
        # 读取文件名列表 这里220341的图片个数不要改变，因为218838_com_three_pair.txt的索引是根据220341_key_list.txt中的位置生成的
        self.pics = np.load('/home/wangxiaopeng/NUS_dataset/220341_conv_pics.npy')
        print 'all pics load over !!'
        # with open(FLAGS.filenames_list_file) as fr:
        #     for i in  fr.readlines():
        #         self.pics.append(self.getimg(os.path.join(FLAGS.imgs_dir,i.strip()+'.jpg')))

        with open(FLAGS.pair_dir) as fr:  # 文件格式：i j k 代表pic名字索引
            pair_files = fr.readlines()

        pair_indexes = []
        pair_len = len(pair_files)
        for i in xrange(1000, pair_len, 1000):
            pair_indexes.append(i)
        self.pair_files = np.array(pair_files)
        np.random.shuffle(self.pair_files)
        self.pair_files = np.split(self.pair_files, pair_indexes)  # 1000行为一个块
        self.pair_files_len = len(self.pair_files)

        print 'read pair_files over !!!'

    # 获取该图片对应的输入向量
    def getimg(self, str1):
        im = Image.open(str1)
        re_img = np.array(im.resize((FLAGS.img_size, FLAGS.img_size)), dtype=np.float32)
        # Subtract off the mean and divide by the variance of the pixels.
        std = np.std(re_img)
        return np.divide(np.subtract(re_img, np.mean(re_img)), std)

    def next_batch(self, batch_size, step):
        # 随机读取false_files中的一个文件（1000行数据），再随机返回batch_size / 8个pairs
        part_files_names = self.pair_files[step % self.pair_files_len]
        np.random.shuffle(part_files_names)
        # np.random.shuffle(self.pair_files)
        i_s = []
        j_s = []
        k_s = []
        for line in part_files_names[:batch_size / 3]:
            # for line in self.pair_files[:batch_size / 3]:
            line = line.strip().split(' ')
            i_s.append(self.pics[int(line[0])])
            j_s.append(self.pics[int(line[1])])
            k_s.append(self.pics[int(line[2])])
        # 一维数组
        return np.concatenate([i_s, j_s, k_s])
