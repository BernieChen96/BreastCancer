# -*- coding: utf-8 -*-
# @Time    : 2022/10/3 21:18
# @Author  : CMM
# @File    : performance_indicators.py
# @Software: PyCharm
import itertools
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import config


def save_classification_report(y_true, y_pred, target_names, save_path):
    # 将分类报告保存至csv文件
    acc_report_df = pd.DataFrame(classification_report(y_true, y_pred, target_names=target_names, output_dict=True)).T
    acc_report_df.iloc[-3, :2] = np.nan
    acc_report_df.iloc[-3, 3] = acc_report_df.iloc[-2, 3]
    # acc_report_df.iloc[-3,2]= np.nan
    acc_report_df.to_csv(save_path)
    return acc_report_df.round(4)


def read_classification_report(path):
    acc_report_df = pd.read_csv(path, index_col=0)
    return acc_report_df.round(4)


def report(y_true, y_pred, target_names):
    print(classification_report(y_true, y_pred, target_names=target_names))


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=55)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
