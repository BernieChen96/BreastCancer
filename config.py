# -*- coding: utf-8 -*-
# @Time    : 2022/10/3 00:47
# @Author  : CMM
# @Site    : 
# @File    : config.py
# @Software: PyCharm
import os

root = os.path.abspath(os.path.dirname(__file__))
random_state = 42
data_root = 'C:/Users/CMM/Desktop/BreaKHis_v1/histology_slides/breast'

# mode: train report
classifier_mode = 'report'
# class: type category
classifier_class = 'category'
classifier_checkpoint = root + f'/classifier/model_parameter_classifier_{classifier_class}_best.pth'
classifier_acc_report = root + f'/report/classifier_{classifier_class}_acc_report.csv'
classifier_confusion_metrix = root + f'/report/classifier_{classifier_class}_confusion_metrix.png'
classifier_result = root + f'/report/classifier_{classifier_class}_result.npy'
if classifier_class == 'type':
    classifier_num_classes = 2
    classifier_class = 0
elif classifier_class == 'category':
    classifier_num_classes = 8
    classifier_class = 1
