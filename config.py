# -*- coding: utf-8 -*-
# @Time    : 2022/10/3 00:47
# @Author  : CMM
# @Site    : 
# @File    : config.py
# @Software: PyCharm
import os
import argparse

import torch
from torchvision.models import DenseNet201_Weights, EfficientNet_B0_Weights

from data.load_data import BreakHisDataset

# ------ Basic configuration ------
root = os.path.abspath(os.path.dirname(__file__))

# ------ gan configuration ------
n_classes = 2
latent_dim = 100
img_size = 32
channels = 3
batch_size = 16
lr = 0.0002
b1 = 0.5
b2 = 0.999
n_epochs = 200
sample_interval = 400


# ------ Basic configuration ------
def get_basic_argument():
    parser = argparse.ArgumentParser("basic configuration")
    parser.add_argument('--root', default=root, help='project root path')
    parser.add_argument('--dataset', default='BreakHis',
                        help='Select dataset.')
    parser.add_argument('--manual_seed', default=42, type=int, help='manual random seed')
    return parser


# ------ classifier configuration ------
def get_classifier_argument():
    parser = get_basic_argument()
    parser.add_argument('--model', default='densenet201',
                        help='Select classification model, densenet201, efficientnet-b0')
    parser.add_argument('--c', default='type', help='Select class to classify. ')
    parser.add_argument('--lr', default=0.0005, type=float, help='learning rate')
    parser.add_argument('--b1', default=0.5, type=float, help='learning rate')
    parser.add_argument('--b2', default=0.999, type=float, help='learning rate')
    parser.add_argument('--epochs', default=20, type=int, help='epochs')
    return parser


def post_classifier_config(opt):
    opt.data_path = get_dataset_path(opt.dataset)
    opt.root = opt.root + '/classifier'
    if opt.dataset == 'BreakHis':
        opt.load_dataset = BreakHisDataset
        if opt.c == 'type':
            opt.n_classes = 2
            opt.label_class = 0
        elif opt.c == 'category':
            opt.n_classes = 8
            opt.label_class = 1
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if opt.model == 'densenet201':
        opt.weights = DenseNet201_Weights.DEFAULT
    elif opt.model == 'efficientnet-b0':
        opt.weights = EfficientNet_B0_Weights.DEFAULT
    else:
        print(f'{opt.model} does not exist.')
        exit(0)
    checkpoint_dir = opt.root + f'/checkpoint/{opt.model}'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    opt.checkpoint = checkpoint_dir + f'/{opt.dataset}_{opt.c}_best.pth'


# ------ metrics configuration ------
def get_metrics_argument():
    parser = get_basic_argument()
    parser.add_argument('--model', default='densenet201',
                        help='Select classification model, densenet201, efficientnet-b0')
    parser.add_argument('--c', default='type', help='Select class to classify. ')
    return parser


def post_metrics_config(opt):
    opt.data_path = get_dataset_path(opt.dataset)
    opt.report_dir = opt.root + f'/report/{opt.model}'
    if opt.model == 'densenet201' or opt.model == 'efficientnet-b0':
        opt.model_type = 'classifier'
        opt.checkpoint = opt.root + f'/classifier/checkpoint/{opt.model}/{opt.dataset}_{opt.c}_best.pth'
        opt.result_path = opt.report_dir + f'/{opt.dataset}_{opt.c}_result.npy'
        opt.prediction_path = opt.report_dir + f'/{opt.dataset}_{opt.c}_prediction.png'
        opt.acc_report_path = opt.report_dir + f'/{opt.dataset}_{opt.c}_acc_report.csv'
        opt.confusion_metrix_path = opt.report_dir + f'/{opt.dataset}_{opt.c}_confusion_metrix.png'
        opt.roc_path = opt.report_dir + f'/{opt.dataset}_{opt.c}_roc.png'
    elif opt.model == '':
        pass
    else:
        print(f'{opt.model} does not exist.')

    if not os.path.exists(opt.checkpoint):
        print('Checkpoint does not exists')
        exit(0)
    if not os.path.exists(opt.report_dir):
        os.makedirs(opt.report_dir)


# ------ data preparation configuration ------
def get_data_preparation_argument():
    parser = get_basic_argument()
    return parser


def post_data_preparation_config(opt):
    opt.data_path = get_dataset_path(opt.dataset)


def get_dataset_path(dataset):
    if dataset == 'BreakHis':
        data_path = 'C:/Users/CMM/Desktop/BreaKHis_v1/histology_slides/breast'
    return data_path
