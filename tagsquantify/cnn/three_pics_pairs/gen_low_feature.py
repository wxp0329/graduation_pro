import numpy as np


def gen_low(fea_file=r'F:\NUS_dataset\Low_Level_Features\Normalized_WT128.dat',
            savefile=r'F:\NUS_dataset\graduate_data\2003_216617_low_feature\wt\216617_ch.txt',
            keylist=r'F:\NUS_dataset\graduate_data\219388_2003_key_list.txt'):
    all_dict = dict()
    num = 0
    with open(r'F:\NUS_dataset\tags\All_Tags.txt',encoding='utf-8_sig') as fr:
        for i in fr.readlines():
            all_dict[i.split(' ', 1)[0]] = num
            num += 1

    all_fea = np.loadtxt(fea_file, dtype=np.int32)
    inds = []
    with open(keylist) as fr:
        for i in fr.readlines():
            inds.append(all_dict[i.strip()])

    np.savetxt(savefile, all_fea[inds], fmt='%d')

if __name__=='__main__':
    gen_low()