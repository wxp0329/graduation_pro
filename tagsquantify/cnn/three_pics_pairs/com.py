# coding:utf-8
from PIL import Image

import datetime
import numpy as np
import shutil
import tensorflow as tf
import os, time
from multiprocessing import cpu_count

import threadpool

# from vgg16train import Vgg16
# from vgg19train import Vgg19

IMAGE_SIZE = 224
IMAGE_DIR = '~NUS_dataset\images_220841'
IMAGE_PATH_2003 = '../169345_0.2_three_pair_pic_name_list.txt'
CHECKPOINT_DIR = r'F:\NUS_dataset\teacher_project_data\checkPointDir'
COM_DIR = './mat_part'
def singleCom():
    root_dir = COM_DIR
    ord = []
    for i in os.listdir(root_dir):
        ord.append(int(i.strip().split('_')[0]))
    ord.sort()
    ls = []
    for i in ord:
        ls.append(np.load(os.path.join(root_dir, str(i) + '_mat.npy')))
    np.save('../vgg16_pool4_169345', np.concatenate(ls))

if __name__=='__main__':
    singleCom()