# coding:utf-8

import numpy as np
from matplotlib import pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['Bitstream Vera Sans']


# X轴，Y轴数据

def acc():
    # beach valley harbor ocean rainbow
    bow = [55.4, 50.3, 55.8, 56.6, 55.1]
    ch = [53.4, 63.3, 60.6, 55, 61.1]
    cm = [57.1, 69.9, 69.2, 58.8, 84.5]
    corr = [58.3, 69.2, 67.6, 59.3, 66.2]
    edh = [65.2, 59.2, 56.3, 71.1, 66.5]
    wt = [65.8, 62.9, 67, 71.3, 74.8]
    cifar = [63.2, 68.4, 70, 65.3, 81]
    vggFeature = [75.1, 76.5, 71.5, 74.3, 79]
    myvgg = [77, 78.8, 85, 75.8, 85.6]

    x = [10, 20, 30, 40, 50]
    x_labels = ['beach', 'valley', 'harbor', 'ocean', 'rainbow']
    # y = [0.3,0.4,2,5,3,4.5,4]
    # y1 = [0.2,0.5,1,4,6,5.5,3]

    plt.figure(figsize=(4, 5))  # 创建绘图对象
    # plt.subplot(121)
    # plt.plot(x, py, color="red", marker='x', ls='dashdot', linewidth=1, label='WSH-CNN')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    # plt.plot(x, py1, color="green", marker='<', linewidth=1, label='DSRH-CNN')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    # plt.plot(x, py2, color="blue", marker='h', linewidth=1, label='BRE-CNN')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    # plt.plot(x, py3, color="magenta", marker='|', linewidth=1, label='BRE')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    # plt.plot(x, py4, color="cyan", marker='d', linewidth=1, label='ITQ')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    # plt.plot(x, py5, color="black", marker='*', linewidth=1, label='LSH')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    # plt.xticks(x, rotation=0)
    # plt.xlabel("#of bits")  # X轴标签
    # plt.ylabel("precision@10")  # Y轴标签
    # plt.subplot(122)
    plt.plot(x, bow, color="b", marker='x', ls='dashdot', linewidth=1, label='BOW')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    plt.plot(x, ch, color="green", marker='<', linewidth=1, label='CH')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    plt.plot(x, cm, color="blue", marker='h', linewidth=1, label='CM')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    plt.plot(x, corr, color="magenta", marker='|', linewidth=1, label='CORR')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    plt.plot(x, edh, color="cyan", marker='d', linewidth=1, label='EDH')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    plt.plot(x, wt, color="black", marker='*', linewidth=1, label='WT')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    plt.plot(x, cifar, color="magenta", marker='>', linewidth=1, label='CIFAR10')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    plt.plot(x, vggFeature, color="cyan", marker='D', linewidth=1, label='VGG16')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    plt.plot(x, myvgg, color="red", marker='o', linewidth=2, label='MY_CNN')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）

    plt.xticks(x, x_labels, rotation=0)

    plt.xlabel("#Catagery")  # X轴标签
    plt.ylabel("#Accuracy (%)")  # Y轴标签
    plt.legend(loc='upper right', ncol=3, bbox_to_anchor=(1.03, 1.16), shadow=True)
    plt.grid(x)
    # plt.title("Precision")  # 图标题
    plt.show()  # 显示图, bbox_to_anchor=(1.29, 1.17)
    # plt.savefig("line.jpg") #保存图

def map():
    # beach valley harbor ocean rainbow
    bow = [38.6, 32.2, 37.8, 39.6,33.8]
    ch = [37.3, 47.9, 43.2, 37.9, 43.1]
    cm = [42.2, 56.2, 52.6, 43.3, 75.1]
    corr = [42.2, 54.5, 51.4, 43.2, 49.8]
    edh = [51.3, 42.9, 39, 57.9, 48.9]
    wt = [51.3, 47.8, 50.3, 58.4, 60.7]
    cifar = [46.4, 54.1, 53, 49.3, 66]
    vggFeature = [63.4, 67, 57.2, 62.3, 66.3]
    myvgg = [66.8, 67, 72.3, 64.5, 76.5]

    x = [10, 20, 30, 40, 50]
    x_labels = ['beach', 'valley', 'harbor', 'ocean', 'rainbow']
    # y = [0.3,0.4,2,5,3,4.5,4]
    # y1 = [0.2,0.5,1,4,6,5.5,3]

    plt.figure(figsize=(4, 5))  # 创建绘图对象
    # plt.subplot(121)
    # plt.plot(x, py, color="red", marker='x', ls='dashdot', linewidth=1, label='WSH-CNN')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    # plt.plot(x, py1, color="green", marker='<', linewidth=1, label='DSRH-CNN')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    # plt.plot(x, py2, color="blue", marker='h', linewidth=1, label='BRE-CNN')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    # plt.plot(x, py3, color="magenta", marker='|', linewidth=1, label='BRE')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    # plt.plot(x, py4, color="cyan", marker='d', linewidth=1, label='ITQ')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    # plt.plot(x, py5, color="black", marker='*', linewidth=1, label='LSH')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    # plt.xticks(x, rotation=0)
    # plt.xlabel("#of bits")  # X轴标签
    # plt.ylabel("precision@10")  # Y轴标签
    # plt.subplot(122)
    plt.plot(x, bow, color="b", marker='x', ls='dashdot', linewidth=1, label='BOW')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    plt.plot(x, ch, color="green", marker='<', linewidth=1, label='CH')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    plt.plot(x, cm, color="blue", marker='h', linewidth=1, label='CM')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    plt.plot(x, corr, color="magenta", marker='|', linewidth=1, label='CORR')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    plt.plot(x, edh, color="cyan", marker='d', linewidth=1, label='EDH')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    plt.plot(x, wt, color="black", marker='*', linewidth=1, label='WT')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    plt.plot(x, cifar, color="magenta", marker='>', linewidth=1, label='CIFAR10')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    plt.plot(x, vggFeature, color="cyan", marker='D', linewidth=1, label='VGG16')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    plt.plot(x, myvgg, color="red", marker='o', linewidth=2, label='MY_CNN')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）

    plt.xticks(x, x_labels, rotation=0)

    plt.xlabel("#Catagery")  # X轴标签
    plt.ylabel("#MAP (%)")  # Y轴标签
    plt.legend(loc='upper right', ncol=3, bbox_to_anchor=(1.03, 1.16), shadow=True)
    plt.grid(x)
    # plt.title("Precision")  # 图标题
    plt.show()  # 显示图, bbox_to_anchor=(1.29, 1.17)
    # plt.savefig("line.jpg") #保存图
def recall():
    # beach valley harbor ocean rainbow
    bow = [5.7, 4.8, 8.3, 5.7, 3.7]
    ch = [4.3,9.7,6.5, 4.4, 3.4]
    cm = [7.9,9.4, 9.6, 7.7, 6.2]
    corr = [6.6, 9.4, 8.2, 6.7, 4.4]
    edh = [9.9, 3.8, 6.9,10.1, 5.8]
    wt = [10.3, 3.9, 6.8, 10.6, 4.7]
    cifar = [9.2, 8, 5.8, 9, 8.2]
    vggFeature = [8.7, 9.1, 7.7, 8.9, 5]
    myvgg = [10.1, 10.7, 10, 10, 7.4]

    x = [10, 20, 30, 40, 50]
    x_labels = ['beach', 'valley', 'harbor', 'ocean', 'rainbow']
    # y = [0.3,0.4,2,5,3,4.5,4]
    # y1 = [0.2,0.5,1,4,6,5.5,3]

    plt.figure(figsize=(4, 5))  # 创建绘图对象
    # plt.subplot(121)
    # plt.plot(x, py, color="red", marker='x', ls='dashdot', linewidth=1, label='WSH-CNN')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    # plt.plot(x, py1, color="green", marker='<', linewidth=1, label='DSRH-CNN')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    # plt.plot(x, py2, color="blue", marker='h', linewidth=1, label='BRE-CNN')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    # plt.plot(x, py3, color="magenta", marker='|', linewidth=1, label='BRE')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    # plt.plot(x, py4, color="cyan", marker='d', linewidth=1, label='ITQ')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    # plt.plot(x, py5, color="black", marker='*', linewidth=1, label='LSH')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    # plt.xticks(x, rotation=0)
    # plt.xlabel("#of bits")  # X轴标签
    # plt.ylabel("precision@10")  # Y轴标签
    # plt.subplot(122)
    plt.plot(x, bow, color="b", marker='x', ls='dashdot', linewidth=1, label='BOW')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    plt.plot(x, ch, color="green", marker='<', linewidth=1, label='CH')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    plt.plot(x, cm, color="blue", marker='h', linewidth=1, label='CM')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    plt.plot(x, corr, color="magenta", marker='|', linewidth=1, label='CORR')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    plt.plot(x, edh, color="cyan", marker='d', linewidth=1, label='EDH')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    plt.plot(x, wt, color="black", marker='*', linewidth=1, label='WT')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    plt.plot(x, cifar, color="magenta", marker='>', linewidth=1, label='CIFAR10')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    plt.plot(x, vggFeature, color="cyan", marker='D', linewidth=1, label='VGG16')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    plt.plot(x, myvgg, color="red", marker='o', linewidth=2, label='MY_CNN')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）

    plt.xticks(x, x_labels, rotation=0)

    plt.xlabel("#Catagery")  # X轴标签
    plt.ylabel("#Recall (%)")  # Y轴标签
    plt.legend(loc='upper right', ncol=3, bbox_to_anchor=(1.03, 1.16), shadow=True)
    plt.grid(x)
    # plt.title("Precision")  # 图标题
    plt.show()  # 显示图, bbox_to_anchor=(1.29, 1.17)
    # plt.savefig("line.jpg") #保存图

def  accMapRecall2003():

    def calcAvgAcc(a):
        return round(np.sum(np.array(a)) * 1.0 / 5, 1)


    bow = [55.4, 50.3, 55.8, 56.6, 55.1]
    ch = [53.4, 63.3, 60.6, 55, 61.1]
    cm = [57.1, 69.9, 69.2, 58.8, 84.5]
    corr = [58.3, 69.2, 67.6, 59.3, 66.2]
    edh = [65.2, 59.2, 56.3, 71.1, 66.5]
    wt = [65.8, 62.9, 67, 71.3, 74.8]
    cifar = [63.2, 68.4, 70, 65.3, 81]
    vggFeature = [75.1, 76.5, 71.5, 74.3, 79]
    myvgg = [77, 78.8, 85, 75.8, 85.6]

    acc = [calcAvgAcc(bow), calcAvgAcc(ch),  calcAvgAcc(cm),  calcAvgAcc(corr),  calcAvgAcc(edh), calcAvgAcc(wt), calcAvgAcc(cifar)
           , calcAvgAcc(vggFeature), calcAvgAcc(myvgg)]

    bow = [38.6, 32.2, 37.8, 39.6, 33.8]
    ch = [37.3, 47.9, 43.2, 37.9, 43.1]
    cm = [42.2, 56.2, 52.6, 43.3, 75.1]
    corr = [42.2, 54.5, 51.4, 43.2, 49.8]
    edh = [51.3, 42.9, 39, 57.9, 48.9]
    wt = [51.3, 47.8, 50.3, 58.4, 60.7]
    cifar = [46.4, 54.1, 53, 49.3, 66]
    vggFeature = [63.4, 67, 57.2, 62.3, 66.3]
    myvgg = [66.8, 67, 72.3, 64.5, 76.5]
    map = [calcAvgAcc(bow), calcAvgAcc(ch), calcAvgAcc(cm), calcAvgAcc(corr), calcAvgAcc(edh), calcAvgAcc(wt),
           calcAvgAcc(cifar)
        , calcAvgAcc(vggFeature), calcAvgAcc(myvgg)]


    bow = [5.7, 4.8, 8.3, 5.7, 3.7]
    ch = [4.3,9.7,6.5, 4.4, 3.4]
    cm = [7.9,9.4, 9.6, 7.7, 6.2]
    corr = [6.6, 9.4, 8.2, 6.7, 4.4]
    edh = [9.9, 3.8, 6.9,10.1, 5.8]
    wt = [10.3, 3.9, 6.8, 10.6, 4.7]
    cifar =[9.2, 8, 5.8, 9, 8.2]
    vggFeature = [8.7, 9.1, 7.7, 8.9, 5]
    myvgg = [10.1, 10.7, 10, 10, 7.4]
    recall  = [calcAvgAcc(bow), calcAvgAcc(ch), calcAvgAcc(cm), calcAvgAcc(corr), calcAvgAcc(edh), calcAvgAcc(wt),
           calcAvgAcc(cifar)
        , calcAvgAcc(vggFeature), calcAvgAcc(myvgg)]


    x = [10, 20, 30, 40, 50,60,70,80,90]
    x_labels = ['BOW', 'CH', 'CM', 'CORR', 'EDH',"WT","CIFAR10","VGG16","MY_CNN"]

    plt.figure(figsize=(4,6.5))  # 创建绘图对象

    plt.plot(x, acc, color="r", marker='x', ls='dashdot', linewidth=1, label='accuracy')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    plt.plot(x, map, color="green", marker='<', linewidth=1, label='MAP')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    plt.plot(x, recall, color="blue", marker='h', linewidth=1, label='recall')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）

    plt.xticks(x, x_labels, rotation=30)

    plt.xlabel("#Feature")  # X轴标签
    plt.ylabel("# (%)")  # Y轴标签
    plt.legend(loc='upper left'  , shadow=True)
    plt.grid(True)
    # plt.title("Precision")  # 图标题
    plt.show()  # 显示图, bbox_to_anchor=(1.29, 1.17)
    # plt.savefig("line.jpg") #保存图


def zhuDraw():
    n_groups = 2

    s = (43.00, 73.42)
    s1 = (45.46, 75.20)
    my = (47.48, 79.54)
    my1 = (50.17, 85.31)

    fig, ax = plt.subplots()

    # index = np.arange(n_groups)
    index = np.arange(n_groups)
    bar_width = 0.2

    opacity = 0.4

    # rects2 = plt.bar(index , s, bar_width, alpha=opacity, color='g', label=u'文献[46]的方法')
    # rects3 = plt.bar(index + bar_width , s1, bar_width, alpha=opacity, color='b', label=u'文献[47]的方法')
    # rects4 = plt.bar(index + bar_width + bar_width, my, bar_width, alpha=opacity, color='r',
    #                  label=u'本文的方法')

    rects2 = plt.bar(index, s, bar_width, alpha=opacity, color='g', label=u'VSM')
    rects3 = plt.bar(index + bar_width, my, bar_width, alpha=opacity, color='b', label=u'VSM+CNN')
    rects4 = plt.bar(index + bar_width + bar_width, s1, bar_width, alpha=opacity, color='g',
                     label=u'VSM+依存关系')
    rects5 = plt.bar(index + bar_width + bar_width + bar_width, my1, bar_width, alpha=opacity, color='r',
                     label=u'本文的检索模型')
    # plt.xlabel(u'方法' )

    plt.ylabel(u'比率')

    # plt.title('filtered candidate set',fontproperties=chinese_font)

    plt.xticks(index + 0.3, (u'F1', u'准确率'))

    plt.ylim(0, 100)

    # plt.legend(loc='upper right', bbox_to_anchor=(1.015,1.05))
    plt.legend(loc='upper right')

    plt.tight_layout()

    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2, height, height, ha='center', va='bottom')
            # rect.set_edgecolor('white')

    # add_labels(rects1)
    add_labels(rects2)
    add_labels(rects3)
    add_labels(rects4)
    add_labels(rects5)

    plt.show()

#2003张测试集在每个特征上的平均准确度，map，recall
def zhu():
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()

            plt.text(rect.get_x() + rect.get_width()/2. , 1.03 * height, "%s" % float(height))

    def calcAvgAcc(a):
        return round(np.sum(np.array(a)) * 1.0 / 5, 1)

    plt.figure(figsize=(5, 6.2))
    plt.xlabel("#test_dataset_avg_acc")
    plt.ylabel("#Accuracy (%)")
    bow = [55.4, 50.3, 55.8, 56.6, 55.1]
    ch = [53.4, 63.3, 60.6, 55, 61.1]
    cm = [57.1, 69.9, 69.2, 58.8, 84.5]
    corr = [58.3, 69.2, 67.6, 59.3, 66.2]
    edh = [65.2, 59.2, 56.3, 71.1, 66.5]
    wt = [65.8, 62.9, 67, 71.3, 74.8]
    cifar = [57.1, 62.4, 60, 59.5, 76]
    vggFeature = [75.1, 76.5, 71.5, 74.3, 79]
    myvgg = [77, 78.8, 85, 75.8, 85.6]
    # plt.title(u"nrrr")
    zuobia = (0, .02, .04, .06, .08, .1, .12, .14, .16)
    plt.xticks(zuobia, ("BOW", "CH", "CM", "CORR", "EDH", "WT", "CIFAR10", "VGG16", "MY_CNN"), rotation=25)
    rect = plt.bar(left=zuobia, height=(
    calcAvgAcc(bow), calcAvgAcc(ch), calcAvgAcc(cm), calcAvgAcc(corr), calcAvgAcc(edh), calcAvgAcc(wt), calcAvgAcc(cifar),
    calcAvgAcc(vggFeature), calcAvgAcc(myvgg)), width=0.01, align="center",
                   yerr=0.000001)

    # plt.legend((rect,), (u"tu",))
    autolabel(rect)

    plt.show()


if __name__ == '__main__':
    acc()
    map()
    recall()
    accMapRecall2003()
    # zhuDraw()
