# coding:utf-8

import numpy as np
from matplotlib import pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['Bitstream Vera Sans']


# X轴，Y轴数据

def map():
    # cnn_bits_acc, cnn_bits_map, cnn_bits_recall low_cnn_avg_acc_map_recall
    data = np.loadtxt(r'F:\NUS_dataset\实验数据结果\低层_cnn_acc_map_recall柱状图数据\cnn_bits_recall.txt',
                      usecols=(i for i in range(1, 6))) * 100
    data=np.transpose(data)
    y = data[0]
    y1 = data[1]
    y2 = data[2]
    y3 = data[3]
    y4 = data[4]
    x = [24, 32, 48, 64, 128]
    plt.figure(figsize=(5, 5))  # 创建绘图对象

    plt.plot(x, y, color="red", marker='x', ls='dashdot', linewidth=1, label='beach')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    plt.plot(x, y1, color="green", marker='<', linewidth=1, label='valley')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    plt.plot(x, y2, color="blue", marker='h', linewidth=1, label='harbor')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    plt.plot(x, y3, color="magenta", marker='|', linewidth=1, label='ocean')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    plt.plot(x, y4, color="cyan", marker='d', linewidth=1, label='rainbow')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）

    plt.xticks(x, rotation=0)
    plt.xlabel("#of bits")  # X轴标签
    plt.ylabel("Recall( % )")  # Y轴标签
    # plt.legend(loc='upper right', bbox_to_anchor=(.6, 1.))
    plt.grid(x)
    # plt.title("Precision")  # 图标题
    plt.show()  # 显示图, bbox_to_anchor=(1.29, 1.17)
    # plt.savefig("line.jpg") #保存图

def avg():
    # cnn_bits_acc, cnn_bits_map, cnn_bits_recall low_cnn_avg_acc_map_recall
    data = np.loadtxt(r'F:\NUS_dataset\实验数据结果\低层_cnn_acc_map_recall柱状图数据\low_cnn_avg_acc_map_recall.txt',
                      usecols=(i for i in range(1, 8))) * 100
    y = data[0]
    y1 = data[1]
    y2 = data[2]
    x = [i for i in range(10,80,10)]
    plt.figure(figsize=(5, 5))  # 创建绘图对象

    plt.plot(x, y, color="red", marker='x', ls='dashdot', linewidth=1, label='Accuracy')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    plt.plot(x, y1, color="green", marker='<', linewidth=1, label='MAP')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    plt.plot(x, y2, color="blue", marker='h', linewidth=1, label='Recall')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）

    plt.xticks(x,('BOW','CH','CM','CORR','EDH','WT','MY_CNN'), rotation=0)
    # plt.xlabel("#of bits")  # X轴标签
    plt.ylabel("#( % )")  # Y轴标签
    plt.legend(loc='upper right', bbox_to_anchor=(.32, 1.01))
    plt.grid(x)
    # plt.title("Precision")  # 图标题
    plt.show()  # 显示图, bbox_to_anchor=(1.29, 1.17)
    # plt.savefig("line.jpg") #保存图

def zhuDraw():
    n_groups = 5
    # low_cnn_map.txt low_cnn_recall.txt lowfeature_cnn_acc.txt
    data = np.loadtxt(r'F:\NUS_dataset\实验数据结果\低层_cnn_acc_map_recall柱状图数据\low_cnn_acc.txt',
                      usecols=(i for i in range(1, 6))) * 100
    bow = data[0]
    ch = data[1]
    cm = data[2]
    corr = data[3]
    edh = data[4]
    wt = data[5]
    mycnn = data[6]

    fig, ax = plt.subplots()

    # index = np.arange(n_groups)
    index = np.arange(n_groups)
    bar_width = 0.12

    opacity = 0.4

    # rects2 = plt.bar(index , s, bar_width, alpha=opacity, color='g', label=u'文献[46]的方法')
    # rects3 = plt.bar(index + bar_width , s1, bar_width, alpha=opacity, color='b', label=u'文献[47]的方法')
    # rects4 = plt.bar(index + bar_width + bar_width, my, bar_width, alpha=opacity, color='r',
    #                  label=u'本文的方法')

    rects2 = plt.bar(index, bow, bar_width, alpha=opacity, color='gray', label=u'1_BOW', )
    rects3 = plt.bar(index + bar_width, ch, bar_width, alpha=opacity, color='b', label=u'2_CH')
    rects4 = plt.bar(index + bar_width + bar_width, cm, bar_width, alpha=opacity, color='deeppink',
                     label=u'3_CM')
    rects5 = plt.bar(index + bar_width + bar_width + bar_width, corr, bar_width, alpha=opacity, color='darkmagenta',
                     label=u'4_CORR')
    rects6 = plt.bar(index + bar_width + bar_width + bar_width + bar_width, edh, bar_width, alpha=opacity,
                     color='darkorchid',
                     label=u'5_EDH' )
    rects7 = plt.bar(index + bar_width + bar_width + bar_width + bar_width + bar_width, wt, bar_width, alpha=opacity,
                     color='y',
                     label=u'6_WT')
    rects1 = plt.bar(index + bar_width + bar_width + bar_width + bar_width + bar_width + bar_width, mycnn, bar_width,
                     alpha=opacity, color='red',
                     label=u'7_MY_CNN')
    # plt.xlabel(u'方法' )

    plt.ylabel(u'#Accuracy( % )')

    # plt.title('filtered candidate set',fontproperties=chinese_font)

    plt.xticks(index + 0.3, (u'beach', u'valley', 'harbor', u'ocean', 'rainbow'))

    plt.ylim(0, 100)

    # plt.legend(loc='upper right', bbox_to_anchor=(1.015,1.05))
    # plt.legend(loc='upper right', bbox_to_anchor=(.61, 1.), ncol=1)

    plt.tight_layout()

    def add_labels(rects,i):
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2, 0, i, ha='center', va='bottom')
            # rect.set_edgecolor('white')

    #
    add_labels(rects2,1)
    add_labels(rects3,2)
    add_labels(rects4,3)
    add_labels(rects5,4)
    add_labels(rects6,5)
    add_labels(rects7,6)
    add_labels(rects1,7)
    plt.show()


if __name__ == '__main__':
    avg()
    # zhuDraw()
