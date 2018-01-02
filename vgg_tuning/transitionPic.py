# coding:utf-8
from PIL import Image

import datetime
import numpy as np
import shutil
import tensorflow as tf
import os, time, sys
from multiprocessing import cpu_count

sys.path.append('/home/wangxiaopeng/graduatePro')
import threadpool

# from tagsquantify.cnn import NUS_layers
# from vgg_tuning import vgg_tuning_net
# from  vgg_tuning import vgg_tuning_net
import vgg16train, vgg_tuning_net

IMAGE_SIZE = 150
IMAGE_DIR = r'F:\NUS_dataset\images_220841'
IMAGE_TRAIN_PATH = r'F:\NUS_dataset\teacher_project_data\test_data\train_com.txt'
IMAGE_PATH_2003 = r'F:\NUS_dataset\teacher_project_data\test_data\test_com.txt'
CHECKPOINT_DIR = r'/home/wangxiaopeng/pool5_tuning_weak_64/model.ckpt-50000'
DATABASE_MAT_DIR = r'F:\NUS_dataset\teacher_project_data\img_transfer_mat\database_mat'
MAT_2003_DIR = r'/home/wangxiaopeng/aa'
COM_DIR = '/home/wangxiaopeng/NUS_dataset/com_dir/'


def get_img(str1):
    im = Image.open(str1)
    re_img = np.array(im.resize((IMAGE_SIZE, IMAGE_SIZE)), dtype=np.float32)
    re_img = (re_img - np.mean(re_img)) / np.std(re_img)
    return re_img


def get_pic_input2output(outleft, outright, path=MAT_2003_DIR):
    global file_paths, w_b
    with tf.Graph().as_default() as g:
        with tf.Session() as sess:

            # 调用模型部分………………………………………………………………………………………………
            # arr = tf.placeholder(dtype=tf.float32, shape=[None, 14, 14, 512])
            arr = tf.placeholder("float", shape=[None, 4096])
            drop = tf.placeholder('bool')
            # vgg16 = vgg16train.Vgg16(vgg16_npy_path=None,
            #                          train_mode=False)
            # arr = vgg16.inference(arr)
            # logit=vgg_tuning_net.vgg_tuning_layer(arr,drop)
            logits = vgg_tuning_net.inference(arr, drop)
            saver = tf.train.Saver()
            saver.restore(sess,
                          CHECKPOINT_DIR)
            # 读取所有图片的.npy文件的个数（为了得到该文件夹中文件的个数）

            # 读取生产的顺序文件（保证最后的向量顺序与该文件里的文件名顺序相同）
            all_pics = file_paths[outleft:outright]
            len_ = len(all_pics)
            i_list = []
            left = 0
            while True:
                right = left + 200
                if right >= len_:
                    i_list.append([left, len_])
                    break
                i_list.append([left, right])
                left = right
                # 开始分批存储转换好的mat 
            for i in i_list:
                affine = sess.run(logits, feed_dict={
                    arr: all_pics[i[0]:i[1]], drop: False})
                # 保存结果
                np.save(os.path.join(path, str(outleft + i[0]) + '_mat'), affine)
                print('第{}块mat保存完毕！'.format(outleft + i[0]))


name = '169345'


def trans_parts():
    global file_paths, name, w_b
    # w_b = np.load(r'/home/wangxiaopeng/tensorflow-vgg_models/vgg16.npy', encoding='latin1').item()
    start = time.time()
    if os.path.exists(MAT_2003_DIR):
        shutil.rmtree(MAT_2003_DIR)

    os.mkdir(MAT_2003_DIR)
    file_paths = np.load(r'/home/wangxiaopeng/graduatePro/vgg16_fc7_active_' + name + '.npy')
    # with open(IMAGE_TRAIN_PATH) as fr:
    #     for i in fr.readlines():
    #         img = get_img(os.path.join(IMAGE_DIR, i.strip() + '.jpg'))
    #         print('读取图片{}完毕！'.format(i))
    #         file_paths.append(img)
    # np.save(r'F:\NUS_dataset\2003_pics',np.array(file_paths).astype(dtype=np.float32))
    print('cpu_count :', cpu_count())
    len_ = len(file_paths)
    i_list = []
    left = 0
    while True:
        right = left + 1000
        if right >= len_:
            i_list.append([left, len_])
            break
        i_list.append([left, right])
        left = right
    n_list = [None for i in range(len(i_list))]
    pool = threadpool.ThreadPool(4)
    requests = threadpool.makeRequests(get_pic_input2output, zip(i_list, n_list))
    [pool.putRequest(req) for req in requests]
    pool.wait()
    print('all of strans_part excute over !!!!!')
    end = time.time()
    print('consume time is :' + str(end - start) + ' seconds!!')


def singleCom():
    global name
    root_dir = r'/home/wangxiaopeng/aa'
    ord = []
    for i in os.listdir(root_dir):
        ord.append(int(i.strip().split('_')[0]))
    ord.sort()
    ls = []
    for i in ord:
        ls.append(np.load(os.path.join(root_dir, str(i) + '_mat.npy')))
    np.save(r'/home/wangxiaopeng/graduatePro/' + name + '_fc7', np.concatenate(ls))


def com(file_id, left, right):
    all_mat_after = []
    for i in file_id[left:right]:
        all_mat_after.append(
            np.load(os.path.join(DATABASE_MAT_DIR, str(i) + '_mat.npy')))

    np.save(os.path.join(COM_DIR, str(left) + '_combine_pic.mat'),
            np.concatenate(all_mat_after))
    print('combination over!!!')


def com_parts():
    if os.path.exists(COM_DIR):
        shutil.rmtree(COM_DIR)

    os.mkdir(COM_DIR)

    print('cpu_count :', cpu_count())
    file_id = []
    for i in range(0, 230000, 100):  # generate files indexes

        if i + 100 >= 218738:
            file_id.append(i)
            break
        file_id.append(i)

    len_ = len(file_id)

    com_i_list = []
    for i in range(0, 230000, 1000):  # split files indexes

        if i + 1000 >= len_:
            com_i_list.append([file_id, i, len_])
            break
        com_i_list.append([file_id, i, i + 1000])
    n_list = [None for ii in range(len(com_i_list))]
    pool = threadpool.ThreadPool(cpu_count())
    requests = threadpool.makeRequests(com, zip(com_i_list, n_list))
    [pool.putRequest(req) for req in requests]
    pool.wait()
    print('all of com excute over !!!!!')


if __name__ == '__main__':
    trans_parts()
    singleCom()
#
