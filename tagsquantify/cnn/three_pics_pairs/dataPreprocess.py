def extract_220341_2003():
    key_dict = dict()
    with open(r'F:\NUS_dataset\tags\tags_after\220341_key_list.dat') as fr:
        num = 0
        for i in fr.readlines():
            key_dict[num] = i.strip()
            num += 1
    key_2003 = []
    with open(r'F:\NUS_dataset\tags\tags_after\2003_key_list.txt') as fr:
        for i in fr.readlines():
            key_2003.append(i.strip())
    fw = open(r'F:\NUS_dataset\tags\tags_after\graduate_data\220341_2003_0.2_three_pair.txt', 'w')
    with open(r'F:\NUS_dataset\tags\tags_after\220341_0.2_three_pair.txt') as fr:
        for i in fr.readlines():
            i_split = i.strip().split(' ')
            s0 = key_dict[int(i_split[0])]
            s1 = key_dict[int(i_split[1])]
            s2 = key_dict[int(i_split[2])]
            if (s0 not in key_2003) and (s1 not in key_2003) and (
                        s2 not in key_2003):
                fw.write(s0 + ' ' + s1 + ' ' + s2 + '\n')
                fw.flush()
    fw.close()


if __name__ == '__main__':
    my_ls = []
    with open(r'F:\NUS_dataset\tags\tags_after\219388_2003_filtered_delEmptyLine.dat',encoding='utf-8') as fr:
        for i in fr.readlines():
            my_ls.append(i.strip().split(sep=' ', maxsplit=1)[0])
    fw = open(r'F:\NUS_dataset\tags\tags_after\graduate_data\219388_2003_0.2_three_pair.txt', 'w')
    with open(r'F:\NUS_dataset\tags\tags_after\graduate_data\220341_2003_0.2_three_pair.txt') as fr:
        for i in fr.readlines():
            i_split = i.strip().split(' ')
            if (i_split[0] in my_ls) and (i_split[1] in my_ls) and (
                        i_split[2] in my_ls):
                fw.write(i)
                fw.flush()
    fw.close()
