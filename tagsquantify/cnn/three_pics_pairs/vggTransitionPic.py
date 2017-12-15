# coding:utf-8
from PIL import Image

import datetime
import numpy as np
import shutil
import tensorflow as tf
import os, time
from multiprocessing import cpu_count

import threadpool

# from tagsquantify.cnn import NUS_layers
from tagsquantify.cnn.three_pics_pairs import Three_net_enforce, explore
from tagsquantify.cnn.vgg_nets.vgg16train import Vgg16
from tagsquantify.cnn.vgg_nets.vgg19train import Vgg19

IMAGE_SIZE = 224
IMAGE_DIR = r'F:\NUS_dataset\images_220841'
IMAGE_TRAIN_PATH = r'F:\NUS_dataset\teacher_project_data\test_data\train_com.txt'
IMAGE_PATH_2003 = r'F:\NUS_dataset\teacher_project_data\2003_key_list.txt'
CHECKPOINT_DIR = r'F:\NUS_dataset\teacher_project_data\checkPointDir'
DATABASE_MAT_DIR = r'F:\NUS_dataset\teacher_project_data\img_transfer_mat\database_mat'
MAT_2003_DIR = r'F:\NUS_dataset\teacher_project_data\img_transfer_mat\2003_mat'
COM_DIR = '/home/wangxiaopeng/NUS_dataset/com_dir/'


def get_img(str1):
    im = Image.open(str1)
    re_img = np.array(im.resize((IMAGE_SIZE, IMAGE_SIZE)), dtype=np.float32)
    re_img = (re_img - np.mean(re_img)) / np.std(re_img)
    return re_img


def singleCom():
    root_dir = r'F:\NUS_dataset\teacher_project_data\img_transfer_mat\database_mat'
    ord = []
    for i in os.listdir(root_dir):
        ord.append(int(i.strip().split('_')[0]))
    ord.sort()
    ls = []
    for i in ord:
        ls.append(np.load(os.path.join(root_dir, str(i) + '_mat.npy')))
    np.save(r'F:\NUS_dataset\teacher_project_data\vgg16_fc7_active_2003', np.concatenate(ls))


def single_gen_mat():
    global file_paths
    file_paths = []
    with open(IMAGE_PATH_2003) as fr:
        for i in fr.readlines():
            # img = get_img(os.path.join(IMAGE_DIR, i.strip() + '.jpg'))
            print('读取图片{}完毕！'.format(i))
            file_paths.append(os.path.join(IMAGE_DIR, i.strip() + '.jpg'))
    len_ = len(file_paths)
    left = 0
    i_list = []
    while True:
        right = left + 1000
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
            vgg16 = Vgg16(vgg16_npy_path=r'F:\NUS_dataset\tensorflow-vgg_models\vgg16.npy')
            logits = vgg16.inference(arr)

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
                    right = left + 50
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
                    np.save(os.path.join(r'F:\NUS_dataset\teacher_project_data\img_transfer_mat\database_mat',
                                         str(lstart + i[0]) + '_mat'), affine)
                    print('第{}块mat保存完毕！'.format(lstart + i[0]))


if __name__ == '__main__':
    singleCom()
    # single_gen_mat()
