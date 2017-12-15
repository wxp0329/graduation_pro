#coding:utf-8
import numpy as np
import os

import shutil


def copyPic(src, tgt):
    with open(tgt, 'wb') as fw:
        with open(src, 'rb') as fr:
            fw.writelines(fr.readlines())


def eval_acc(step,mat_121, mat_1000,acc_file):
    write_acc = open(os.path.join(acc_file,'_acc.txt'), 'a')
    global picname_121, picname_1000, pic_121_labels, pic_1000_labels
    sim_count = 50
    # mat_121 = np.where(mat_121 >= 0.5, 1., 0.)
    # mat_1000 = np.where(mat_1000 >= 0.5, 1., 0.)
    pic_121_labels = np.loadtxt(r'F:\NUS_dataset\teacher_project_data\test_data\test_com_81Labels.txt', dtype=np.int)
    pic_1000_labels = np.loadtxt(r'F:\NUS_dataset\teacher_project_data\test_data\train_com_81Labels.txt', dtype=np.int)
    acc_ls = []
    pic_num = 0
    for i in mat_121:
        dist=np.sum(np.square(np.subtract(i, mat_1000)), axis=1)#欧式距离
        # dist = np.sum(np.where(np.subtract(i, mat_1000) != 0, 1, 0), axis=1)#汉明距离
        min_dist_ls = np.argsort(dist)[:sim_count]
        acc_com = acc(pic_num, min_dist_ls)
        acc_ls.append(acc_com)  # compute acc
    # print('all avg acc is : {}'.format(np.mean(acc_ls)))
    write_acc.write(str(step)+' '+str(np.mean(acc_ls))+'\n')
    write_acc.close()
def acc(img1_ind, min_dist_imgs_ind):
    global picname_121, picname_1000, pic_121_labels, pic_1000_labels

    pic_121_Lb = pic_121_labels[img1_ind]
    pic_1000_Lb = pic_1000_labels[min_dist_imgs_ind]  
    #
    sum_aix1 = np.sum(np.multiply(pic_121_Lb, pic_1000_Lb), axis=1)
    match_count = len(sum_aix1[np.where(sum_aix1 != 0)]) * 1. / len(sum_aix1)
    return match_count

if __name__ == '__main__':
    write_explore_pics()
