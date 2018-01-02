# coding:utf-8
from PIL import Image

import datetime
import numpy as np
import shutil
import tensorflow as tf
import os, time
from multiprocessing import cpu_count

import threadpool

from vgg16train import Vgg16
from vgg19train import Vgg19

IMAGE_SIZE = 224
IMAGE_DIR = '/home/wangxiaopeng/NUS_dataset/images_220841'
IMAGE_PATH_2003 = '/home/wangxiaopeng/graduatePro/2000_frequent_key_list.txt'#F:\NUS_dataset\graduate_data\169345_0.2_three_pair_pic_name_list.txt
MAT_2003_DIR = '/home/wangxiaopeng/aa'


def get_img(str1):
    im = Image.open(str1)
    re_img = np.array(im.resize((IMAGE_SIZE, IMAGE_SIZE)), dtype=np.float32)
    re_img = (re_img - np.mean(re_img)) / np.std(re_img)
    return re_img


def singleCom():
    root_dir = MAT_2003_DIR
    ord = []
    for i in os.listdir(root_dir):
        ord.append(int(i.strip().split('_')[0]))
    ord.sort()
    ls = []
    for i in ord:
        ls.append(np.load(os.path.join(root_dir, str(i) + '_mat.npy')))
    np.save('/home/wangxiaopeng/graduatePro/vgg19_fc7_2000' , np.concatenate(ls))


def single_gen_mat():
    global file_paths
    file_paths = []
    if os.path.exists(MAT_2003_DIR):
        shutil.rmtree(MAT_2003_DIR)

    os.mkdir(MAT_2003_DIR)
    with open(IMAGE_PATH_2003) as fr:
        for i in fr.readlines():
            # img = get_img(os.path.join(IMAGE_DIR, i.strip() + '.jpg'))
            i=i.strip().replace('\xef\xbb\xbf', '')
            print('读取图片{}完毕！'.format(i))
            file_paths.append(os.path.join(IMAGE_DIR, i + '.jpg'))
            # file_paths.append(img)
    len_ = len(file_paths)
    left = 0
    i_list = []
    while True:
        right = left + 100
        if right >= len_:
            i_list.append([left, len_])
            break
        i_list.append([left, right])
        left = right
    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 1.
    with tf.Graph().as_default() as g:
        with tf.Session(config=config) as sess:

            # 调用模型部分………………………………………………………………………………………………
            arr = tf.placeholder("float", [None, IMAGE_SIZE, IMAGE_SIZE, 3])
            vgg19= Vgg19(vgg19_npy_path='/home/wangxiaopeng/tensorflow-vgg_models/vgg19.npy')
            logits = vgg19.inference(arr)
            sess.run(tf.global_variables_initializer())
            # 读取生产的顺序文件（保证最后的向量顺序与该文件里的文件名顺序相同）
            for i in i_list:
                lstart = i[0]
                lend = i[1]
                num = 1
                all_pics = []
                for i in file_paths[lstart:lend]:
                    img_mat = get_img(i)
                    all_pics.append((img_mat - np.mean(img_mat)) / np.std(img_mat))
                    print(num)
                    num += 1
                len_ = len(all_pics)
                i_list = []
                left = 0
                while True:
                    right = left + 30
                    if right >= len_:
                        i_list.append([left, len_])
                        break
                    i_list.append([left, right])
                    left = right
                    # 开始分批存储转换好的mat
                for i in i_list:
                    affine = sess.run(logits, feed_dict={
                        arr: all_pics[i[0]:i[1]]})
                    # 保存结果
                    np.save(os.path.join(MAT_2003_DIR,
                                         str(lstart + i[0]) + '_mat'), affine)
                    print('第{}块mat保存完毕！'.format(lstart + i[0]))


if __name__ == '__main__':
    single_gen_mat()
    singleCom()
