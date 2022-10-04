# -*- coding: utf-8 -*-
# @Time    : 2022/10/3 00:47
# @Author  : CMM
# @Site    : 
# @File    : config.py
# @Software: PyCharm
import os

# ------ Basic configuration ------
root = os.path.abspath(os.path.dirname(__file__))
random_state = 42
data_root = 'C:/Users/CMM/Desktop/BreaKHis_v1/histology_slides/breast'

# ------ classifier configuration ------
# mode: train report
classifier_mode = 'report'
# class: type category
classifier_class = 'type'
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

# ------ gan configuration ------
n_classes = classifier_num_classes
latent_dim = 100
img_size = 32
channels = 3
batch_size = 16
lr = 0.0002
b1 = 0.5
b2 = 0.999
n_epochs = 200
sample_interval = 400
