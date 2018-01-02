import threadpool
from PIL import Image


def trans_ind():
    name_dict = dict()
    with open(r'F:\NUS_dataset\graduate_data\219388_2003_key_list.dat') as fr:
        num = 0
        for i in fr.readlines():
            name_dict[i.strip()] = str(num)
            num += 1
    fw = open(r'F:\NUS_dataset\graduate_data\219388_2003_indexes_0.2_three_pair.dat', 'w')
    with open(r'F:\NUS_dataset\graduate_data\219388_2003_0.2_filename_three_pair.txt') as fr:
        for i in fr.readlines():
            line_split = i.strip().split(' ')
            fw.write(name_dict[line_split[0]] + ' ' + name_dict[line_split[1]] + ' ' + name_dict[line_split[2]] + '\n')
    fw.close()

import numpy as np
import os
def getimg(str1):
    im = Image.open(str1)
    re_img = np.array(im.resize((100, 100)), dtype=np.float32)
    # Subtract off the mean and divide by the variance of the pixels.
    return (re_img - np.mean(re_img)) / np.std(re_img)
def run(l,r):
    global ps
    a=[]
    for i in ps[l:r]:
        a.append(getimg(os.path.join(r'F:\NUS_dataset\images_220841', i.strip() + '.jpg')))
    np.save(os.path.join(r'f:\aa',str(l)), np.array(a))
    print('{} is saved!!'.format(l))
def part():
    global ps
    fr=open(r'F:\NUS_dataset\graduate_data\219388_2003_0.2_three_pair_pic_name_list.txt')
    ps=fr.readlines()
    len_=len(ps)
    i_list = []
    for i in range(0, 230000, 1000):  # 3000000 行大约50M数据

        if i + 3000 >= len_:
            i_list.append([i, len_ ])
            break
        i_list.append([ i, i + 3000])
    n_list = [None for i in range(len(i_list))]
    pool = threadpool.ThreadPool(os.cpu_count())
    requests = threadpool.makeRequests(run, zip(i_list, n_list))
    [pool.putRequest(req) for req in requests]
    pool.wait()

def singleCom():
    root_dir = r'f:\aa'
    ord = []
    for i in os.listdir(root_dir):
        ord.append(int(i.strip().split('.')[0]))
    ord.sort()
    ls = []
    for i in ord:
        ls.append(np.load(os.path.join(root_dir, str(i) + '.npy')))
    np.save(r'F:\NUS_dataset\graduate_data\219388_2003_pic_norm_mat', np.concatenate(ls))
def ha():
    paths = []
    root_dir = r'F:\NUS_dataset\Groundtruth\AllLabels'
    for i in os.listdir(root_dir):
        paths.append(i.strip())
    paths.sort()
    fw=open(r'F:\NUS_dataset\teacher_project_data\81_labels_sort.txt','w')
    num=0
    for i in paths:
        fw.write(str(num)+'_'+i.split('_')[1].split('.')[0]+'\n')
        num+=1
    fw.close()
def ex_clazz(class_num,class_name):
    name_ls=np.loadtxt(r'F:\NUS_dataset\graduate_data\2003_key_list.txt')
    a=np.loadtxt(r'F:\NUS_dataset\graduate_data\2003_81_labels_mat.txt',dtype=np.int32)
    b=a[:,class_num]
    w_ls=np.where(b==1)[0].tolist()
    np.savetxt('F:\\NUS_dataset\\graduate_data\\2003中四种类别的lables和keylist\\'+class_name+'_labels.txt',a[w_ls],fmt='%d')
    np.savetxt('F:\\NUS_dataset\\graduate_data\\2003中四种类别的lables和keylist\\'+class_name+'_picname.txt',name_ls[w_ls],fmt='%d')
def ex_fea(class_,fea_name):
    class_num=int(class_.split('_')[0])
    class_name=class_.split('_')[1]
    a = np.loadtxt(r'F:\NUS_dataset\graduate_data\2003_81_labels_mat.txt', dtype=np.int32)
    fea_ls = np.loadtxt('F:\\NUS_dataset\\graduate_data\\2003_216617_low_feature\\'+fea_name+'\\2003_'+fea_name+'.txt')
    b = a[:,class_num]
    w_ls = np.where(b == 1)[0].tolist()
    np.savetxt('F:\\NUS_dataset\\graduate_data\\2003中四种类别的lables和keylist\\'+fea_name+'\\'+class_name+'_'+fea_name+'.txt', fea_ls[w_ls],
               fmt='%d')


def cc():
    a = []
    with open(r'F:\NUS_dataset\teacher_project_data\81_labels_sort.txt') as fr:
        a = fr.readlines()
    b = ['bow', 'ch', 'cm', 'corr', 'edh', 'wt']
    for i in b:
        for j in a:
            ex_fea(j.strip(), i)


def test_pair_name():
    a = np.loadtxt(r'F:\NUS_dataset\teacher_project_data\216617_81_labels_mat.txt', dtype=np.int32)
    label = np.loadtxt(r'F:\NUS_dataset\teacher_project_data\81_labels_sort.txt', dtype=np.string_)
    lis = np.loadtxt(r'F:\NUS_dataset\teacher_project_data\219388_2003_sparsecoding_bow_keylist.txt', dtype=np.string_)
    bei = 216141
    t1 = 216162
    t2 = 216176
    t3 = 216250
    b = label[np.where(np.multiply(a[bei], a[t1]) > 0)[0]]
    b1 = label[np.where(np.multiply(a[bei], a[t2]) > 0)[0]]
    b2 = label[np.where(np.multiply(a[bei], a[t3]) > 0)[0]]
    c = label[np.where(a[bei] > 0)[0]]
    c1 = label[np.where(a[t1] > 0)[0]]
    c2 = label[np.where(a[t2] > 0)[0]]
    c3 = label[np.where(a[t3] > 0)[0]]
    print(b)
    print(b1)
    print(b2)
    print('bei labels: {},filename is {}'.format(c, lis[bei]))
    print('t1 labels: {},filename is {}'.format(c1, lis[t1]))
    print('t2 labels: {},filename is {}'.format(c2, lis[t2]))
    print('t3 labels: {},filename is {}'.format(c3, lis[t3]))

def comTreePair():
    true_dict=dict()
    false_dict=dict()
    true_dir=r'F:\NUS_dataset\teacher_project_data\sim_or_not_pair\yes'
    for filename in os.listdir(true_dir):
        with open(os.path.join(true_dir,filename)) as fr:
            for i in fr.readlines():
                line=i.strip().split(' ')
                true_dict[line[0]]=line[1:]

    false_dir=r'F:\NUS_dataset\teacher_project_data\sim_or_not_pair\no'
    for filename in os.listdir(false_dir):
        with open(os.path.join(false_dir,filename)) as fr:
            for i in fr.readlines():
                line=i.strip().split(' ')
                false_dict[line[0]]=line[1:]
    fw=open(r'F:\NUS_dataset\teacher_project_data\216617_three_pair_index.txt','w')
    num=0
    for i in true_dict:
        true_list=true_dict[i]
        false_list=false_dict[i]
        len_=min([len(true_list),len(false_list)])
        for j in range(len_):
            fw.write(i+' '+true_list[j]+' '+false_list[j]+'\n')
        fw.flush()
        num+=1
        print('第{}个i已经写完'.format(num))
    fw.close()
def jiao():
    a=set()
    with open(r'F:\NUS_dataset\graduate_data\219388_2003_0.2_filename_three_pair.txt') as fr:
        for i in fr.readlines():
            for j in i.split(' '):
                a.add(j)
    b=set()
    with open(r'F:\NUS_dataset\graduate_data\2003_key_list.txt') as fr:
        for i in fr.readlines():
            a.add(i.strip())
    print(a.intersection(b))

def ex_169345_fc():
    a=dict()
    with open(r'F:\NUS_dataset\teacher_project_data\219388_2003_sparsecoding_bow_keylist.txt') as fr:
        num=0
        for i in fr.readlines():
            a[i.strip()]=num
            num+=1
    b=[]
    with open(r'F:\NUS_dataset\graduate_data\169345_0.2_three_pair_pic_name_list.txt') as fr:
        for i in fr.readlines():
            b.append(a[i.strip()])
    d=np.load(r'F:\NUS_dataset\teacher_project_data\vgg16_fc7_active_216617.npy')[b]
    print(d.shape)
    np.save(r'F:\NUS_dataset\graduate_data\169345_vgg16_fc',d)
def aa():
    small=np.loadtxt(r'F:\NUS_dataset\teacher_project_data\2003_81_labels_mat.txt',dtype=np.int32)
    big = np.loadtxt(r'F:\NUS_dataset\teacher_project_data\216617_81_labels_mat.txt', dtype=np.int32)
    sum=0
    fw=open(r'test.txt','w')
    for i in small:
        result=np.sum(np.multiply(i,big),axis=1)
        count=len(np.where(result!=0)[0].tolist())
        if count < 500:
            fw.write(str(sum)+' '+str(count)+' ##############')
            print(str(sum)+' '+str(count)+' ##############')
        else:
            fw.write(str(sum) + ' ' + str(count))
            print(str(sum) + ' ' + str(count))
        sum+=1
    fw.close()


if __name__=='__main__':
    mydict = np.load(r'F:\NUS_dataset\graduate_data\img_transfer_mat\mat_part\0_mat.npy' )
    print(mydict.shape)