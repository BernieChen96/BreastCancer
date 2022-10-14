# -*- coding: utf-8 -*-
# @Time    : 2022/10/14 9:49
# @Author  : CMM
# @File    : test.py
# @Software: PyCharm
import torch

from config import get_dataset_path
from data.image_processing import show, dataset_detail
from data.load_data import cifar10_dataset

path = get_dataset_path('cifar10')
torch.manual_seed(42)
# data_path = 'C:/Users/CMM/Desktop/BreaKHis_v1/histology_slides/breast'
# show(cifar10_dataset, params={"train": True, "data_path": path})
# show(BreakHisDataset, params={"data_path": "C:/Users/CMM/Desktop/BreaKHis_v1/histology_slides/breast",
#                               "train": True})
dataset_detail(cifar10_dataset, params={"train": True, "data_path": path})
