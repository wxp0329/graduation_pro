import numpy as np
import os

import shutil


def copyPic(src, tgt):
    with open(tgt, 'wb') as fw:
        with open(src, 'rb') as fr:
            fw.writelines(fr.readlines())


def write_explore_pics(
    class_,clazz
        ):
    global picname_121, picname_1000, pic_121_labels, pic_1000_labels
    # class_=['beach','flowers','mountain','valley']
    # clazz=['bow','ch','cm','corr','edh','wt']#beach flowers mountain valley

    mat_121_file = os.path.join(os.path.join(r'F:\NUS_dataset\graduate_data\2003中四种类别的lables和keylist',clazz),class_+'_'+clazz+'.txt')
    #r'F:\NUS_dataset\graduate_data\2003_216617_low_feature\bow\2003_bow.txt'
    mat_1000_file = os.path.join(os.path.join(r'F:\NUS_dataset\graduate_data\2003_216617_low_feature',clazz),'216617_'+clazz+'.txt')
    sim_count = 500
    img_dir = r'F:\NUS_dataset\images_220841'
    exploreDir = r'F:\NUS_dataset\teacher_project_data\exploreImgs'
    # if os.path.exists(exploreDir):
    #     shutil.rmtree(exploreDir)
    # os.mkdir(exploreDir)
    picname_121 = np.loadtxt(os.path.join(r'F:\NUS_dataset\graduate_data\2003中四种类别的lables和keylist',class_+'_picname.txt'), dtype=np.string_)
    picname_1000 = np.loadtxt(r'F:\NUS_dataset\graduate_data\219388_2003_key_list.txt', dtype=np.string_)
    # 加载测试集图片和训练集图片的81个标签矩阵
    pic_121_labels = np.loadtxt(os.path.join(r'F:\NUS_dataset\graduate_data\2003中四种类别的lables和keylist',class_+'_labels.txt'), dtype=np.int)
    pic_1000_labels = np.loadtxt(r'F:\NUS_dataset\graduate_data\219388_2003_81_labels_mat.txt', dtype=np.int)
    # 加载模型转换完的图片向量
    mat_121 = np.loadtxt(mat_121_file)
    mat_1000 = np.loadtxt(mat_1000_file)
    acc_ls = []
    map_ls=[]
    recall_ls=[]
    pic_num = 0
    for i in mat_121:
        picname = str(picname_121[pic_num]).strip('b').replace('\'', '')
        i_dir = os.path.join(exploreDir, picname)
        # os.mkdir(i_dir)
        # copyPic(os.path.join(img_dir, picname + '.jpg'), os.path.join(i_dir, '000_' + picname + '.jpg'))
        sum = 0
        min_dist_ls = np.argsort(np.sum(np.square(np.subtract(i, mat_1000)), axis=1))[:sim_count]
        pic_121_Lb = pic_121_labels[pic_num]  # 被查询图片对应的81个标签集
        pic_1000_Lb = pic_1000_labels[min_dist_ls]  # 查询到的图片集所对应的81个标签矩阵
        sum_aix1 = np.sum(np.multiply(pic_121_Lb, pic_1000_Lb), axis=1)
        acc_com = acc(sum_aix1)
        map_com=MAP(sum_aix1)
        recall_com=recall(pic_121_Lb,sum_aix1)
        if acc_com == 0.:
            map_com=0.
            recall_com=0.
        print('第{}个图片{}的acc为：{},map 为：{}，recall为：{}。'.format(pic_num+1,picname, acc_com,map_com,recall_com))
        acc_ls.append(acc_com)  # compute acc
        map_ls.append(map_com)
        recall_ls.append(recall_com)
        for j in picname_1000[min_dist_ls]:
            picname1 = str(j).strip('b').replace('\'', '')
            # copyPic(os.path.join(img_dir, picname1 + '.jpg'), os.path.join(i_dir, str(sum) + '_' + picname1 + '.jpg'))
            sum += 1
        pic_num += 1
    name=os.path.split(mat_121_file)[1].split('_')[1]
    fw=open(os.path.join(r'C:\Users\Administrator\Desktop\aa',class_+'_'+clazz+'eval.txt'),'w')
    fw.write(class_+'_'+clazz+' avg acc is : {}\n'.format(np.mean(acc_ls)))
    fw.write(class_+'_'+clazz+' avg map is : {}\n'.format(np.mean(map_ls)))
    fw.write(class_+'_'+clazz+' avg recall is : {}\n'.format(np.mean(recall_ls)))
    fw.close()


def acc(sum_aix1):
    match_count = len(sum_aix1[np.where(sum_aix1 != 0)]) * 1. / len(sum_aix1)
    return match_count

def MAP(sum_aix1):
    sum_aix1 = np.reshape(sum_aix1, [-1])
    aim_inds = np.where(sum_aix1 != 0)[0]+1
    if aim_inds.shape[0] == 0:
        return 0
    c = np.array([i for i in range(1, len(aim_inds) + 1)])
    return np.mean(np.divide(c,aim_inds))


def recall(pic_121_Lb, sum_aix1):
    global picname_121, picname_1000, pic_121_labels, pic_1000_labels
    all_count = np.sum(np.multiply(pic_121_Lb, pic_1000_labels), axis=1)
    return len(sum_aix1[np.where(sum_aix1 != 0)]) * 1. / len(all_count[np.where(all_count != 0)])


def gen_part_label(filename=r'F:\NUS_dataset\teacher_project_data\test_data\train_com.txt'):
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
    with open(r'F:\NUS_dataset\teacher_project_data\test_data\test_com.txt') as fr:
        for i in fr.readlines():
            label_ind.append(all_dict[i.strip()])

    np.savetxt(r'F:\NUS_dataset\teacher_project_data\test_data\test_com_81Labels.txt', all_col[label_ind], fmt='%s')


def four_class_acc_map_recall():
    class_ = ['beach', 'flowers', 'mountain', 'valley']
    clazz = ['bow', 'ch', 'cm', 'corr', 'edh', 'wt']
    for i in clazz:
        for j in class_:
            write_explore_pics(
                j, i
            )


if __name__ == '__main__':
    four_class_acc_map_recall()
