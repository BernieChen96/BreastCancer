# -*- coding: utf-8 -*-
# @Time    : 2022/10/3 21:18
# @Author  : CMM
# @File    : performance_indicators.py
# @Software: PyCharm
import itertools
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, auc, roc_curve
from sklearn.preprocessing import label_binarize


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


def plot_confusion_matrix(y_true,
                          y_pred,
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          save_path=None):
    cm = confusion_matrix(y_true, y_pred)
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
    plt.savefig(save_path)
    plt.show()


def plot_roc(y_true, y_pred):
    # roc_log = roc_auc_score(y_true, y_pred)
    false_positive_rate, true_positive_rate, threshold = roc_curve(y_true,
                                                                   y_pred)
    area_under_curve = auc(false_positive_rate, true_positive_rate)
    print(area_under_curve)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.plot(false_positive_rate, true_positive_rate, label='AUC = {:.3f}'.format(area_under_curve))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()
    # plt.savefig(ROC_PLOT_FILE, bbox_inches='tight')


def plot_roc_mutil(y_true, y_pred_prob, labels):
    print(labels)
    classes = list(range(len(labels)))
    y_true = label_binarize(y_true, classes=classes)
    y_pred_prob = np.array(y_pred_prob)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(labels)):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 8))
    colors = ['red', 'orange', 'yellow', 'green', 'cyan',
              'blue', 'purple', 'pink', 'magenta', 'brown']
    for i, color in zip(range(len(labels)), colors):
        plt.plot(fpr[i], tpr[i], color=color,
                 label='ROC curve of class {0} (area = {1:0.3f})'
                       ''.format(labels[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for multi-class data')
    plt.legend(loc="lower right")
    plt.show()
