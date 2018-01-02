import numpy as np
import os

import time


def write_explore_pics(mytype='hanming'):
    sim_count = 500
    mat_121_file = r'F:\NUS_dataset\graduate_data\img_transfer_mat\2000_64_have.npy'
    mat_1000_file = r'F:\NUS_dataset\graduate_data\img_transfer_mat\169345_64_have.npy'
    # ft='corr'
    # mat_121_file = r'F:\NUS_dataset\graduate_data\2000_169345_low_feature\2000_'+ft+'.txt'
    # mat_1000_file = r'F:\NUS_dataset\graduate_data\2000_169345_low_feature\169345_'+ft+'.txt'
    global picname_121, picname_1000, pic_121_labels, pic_1000_labels

    img_dir = r'F:\NUS_dataset\images_220841'
    exploreDir = r'F:\NUS_dataset\graduate_data\exploreImgs'
    picname_121 = np.loadtxt(open(r'F:\NUS_dataset\graduate_data\2000_frequent_key_list.txt', encoding='utf-8_sig'),
                             dtype=np.string_)
    picname_1000 = np.loadtxt(r'F:\NUS_dataset\graduate_data\169345_0.2_three_pair_pic_name_list.txt',
                              dtype=np.
                              string_)
    # 加载测试集图片和训练集图片的81个标签矩阵
    pic_121_labels = np.loadtxt(r'F:\NUS_dataset\graduate_data\2000_frequent_81_labels_mat.txt', dtype=np.int)
    pic_1000_labels = np.loadtxt(r'F:\NUS_dataset\graduate_data\169345_81_labels_mat.txt', dtype=np.int)
    # # 加载模型转换完的图片向量
    mat_121 = np.load(mat_121_file)
    mat_1000 = np.load(mat_1000_file)
    # mat_121 = np.loadtxt(mat_121_file)
    # mat_1000 = np.loadtxt(mat_1000_file)
    if mytype == 'hanming':
        print('hanming 计算。。。。。。。。。')
        mat_121 = np.where(mat_121 > 0., 1., 0.)
        mat_1000 = np.where(mat_1000 > 0., 1., 0.)
    acc_ls = []
    map_ls = []
    recall_ls = []
    pic_num = 0
    start = time.time()
    for i in mat_121:
        # if pic_num in [1387,1740,1963,519,568,836]:
        #     pic_num += 1
        #     continue
        picname = str(picname_121[pic_num]).strip('b').replace('\'', '')
        # os.mkdir(i_dir)
        # copyPic(os.path.join(img_dir, picname + '.jpg'), os.path.join(i_dir, '000_' + picname + '.jpg'))
        sum = 0
        if mytype == 'oushi':
            dist = np.sum(np.square(np.subtract(i, mat_1000)), axis=1)  # 欧式距离
        else:
            dist = np.sum(np.where(np.subtract(i, mat_1000) != 0, 1, 0), axis=1)  # 汉明距离
        min_dist_ls = np.argsort(dist)[:sim_count]
        pic_121_Lb = pic_121_labels[pic_num]  # 被查询图片对应的81个标签集
        pic_1000_Lb = pic_1000_labels[min_dist_ls]  # 查询到的图片集所对应的81个标签矩阵
        sum_aix1 = np.sum(np.multiply(pic_121_Lb, pic_1000_Lb), axis=1)
        acc_com = acc(sum_aix1)
        map_com = MAP(sum_aix1)
        recall_com = recall(pic_121_Lb, sum_aix1)
        print('第{}个图片{}的acc为：{},map为{},recall为{}。'.format(pic_num + 1, picname, acc_com, map_com, recall_com))
        acc_ls.append(acc_com)  # compute acc
        map_ls.append(map_com)
        recall_ls.append(recall_com)
        for j in picname_1000[min_dist_ls]:
            picname1 = str(j).strip('b').replace('\'', '')
            # copyPic(os.path.join(img_dir, picname1 + '.jpg'), os.path.join(i_dir, str(sum) + '_' + picname1 + '.jpg'))
            sum += 1
        pic_num += 1
    print(
        mytype + '[ 12 ]--- acc_ls len is {},all avg acc is : {},all avg map is :{}, all avg recall is: {},time is {}.'.format(
            len(acc_ls), np.mean(acc_ls), np.mean(map_ls), np.mean(recall_ls), time.time() - start))


def acc(sum_aix1):
    '''img1_ind:所查询图片在picname_121的索引
    min_dist_imgs_ind:查询到的距离最小的在picname_1000中的索引集
    '''

    match_count = len(sum_aix1[np.where(sum_aix1 != 0)]) * 1. / len(sum_aix1)
    return match_count


def MAP(sum_aix1):
    sum_aix1 = np.reshape(sum_aix1, [-1])
    aim_inds = np.where(sum_aix1 != 0)[0] + 1.
    if aim_inds.shape[0] == 0:
        return 0
    c = np.array([i for i in range(1, len(aim_inds) + 1)]).astype(np.float32)
    return np.mean(np.divide(c, aim_inds))


def recall(pic_121_Lb, sum_aix1):
    global picname_121, picname_1000, pic_121_labels, pic_1000_labels
    all_count = np.sum(np.multiply(pic_121_Lb, pic_1000_labels), axis=1)
    return len(sum_aix1[np.where(sum_aix1 != 0)]) * 1. / len(all_count[np.where(all_count != 0)])


def run():
    pass


def mult():
    pass


def gen_part_label(filename=r'F:\NUS_dataset\graduate_data\test_data\train_com.txt'):
    paths = []
    root_dir = r'F:\NUS_dataset\Groundtruth\AllLabels'
    for i in os.listdir(root_dir):
        paths.append(i.strip())
    paths.sort()
    all_label = []
    for i in paths:
        all_label.append(np.reshape(np.loadtxt(os.path.join(root_dir, i), dtype=np.int), [-1, 1]))
    all_col = np.concatenate(all_label, axis=1)

    all_dict = dict()
    with open(r'F:\NUS_dataset\tags\All_Tags.txt', encoding='utf-8') as fr:
        num = 0
        for i in fr.readlines():
            all_dict[i.strip().split(sep=' ', maxsplit=1)[0]] = num
            num += 1
    label_ind = []
    with open(r'F:\NUS_dataset\graduate_data\test_data\test_com.txt') as fr:
        for i in fr.readlines():
            label_ind.append(all_dict[i.strip()])

    np.savetxt(r'F:\NUS_dataset\graduate_data\test_data\test_com_81Labels.txt', all_col[label_ind], fmt='%s')


if __name__ == '__main__':
    write_explore_pics()
