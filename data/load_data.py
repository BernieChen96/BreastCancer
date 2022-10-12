# -*- coding: utf-8 -*-
# @Time    : 2022/10/2 13:32
# @Author  : CMM
# @Site    : 
# @File    : load_data.py
# @Software: PyCharm
import csv
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from data.image_processing import display_some_images


class BreakHisDataset(Dataset):  # 需要继承data.Dataset
    def __init__(self, data_path, train, repeat=1, transform=None, ):
        """
        :param data_path 数据根目录
        :param train True or False 是否为训练数据
        :param repeat: 所有样本数据重复次数，默认循环一次，当repeat为None时，表示无限循环<sys.maxsize
        """
        # TODO
        # 1. Initialize file path or list of file names.
        if train:
            filename = data_path + '/train.csv'
        else:
            filename = data_path + '/test.csv'
        self.image_label_list = self.read_file(filename)
        self.len = len(self.image_label_list)
        self.repeat = repeat
        self.transform = transform
        self.encode_type = np.load(data_path + '/encode_type.npy', allow_pickle='TRUE').item()
        self.encode_category = np.load(data_path + '/encode_category.npy', allow_pickle='TRUE').item()
        self.decode_type = dict(zip(self.encode_type.values(), self.encode_type.keys()))
        self.decode_category = dict(zip(self.encode_category.values(), self.encode_category.keys()))

    def __getitem__(self, i):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        # 这里需要注意的是，第一步：read one data，是一个data
        index = i % self.len
        # print("i={},index={}".format(i, index))
        image_path, labels = self.image_label_list[index]
        img = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        labels = [self.encode_type[labels[0]], self.encode_category[labels[1]]]
        return img, labels

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        if self.repeat is None:
            data_len = 10000000
        else:
            data_len = len(self.image_label_list) * self.repeat
        return data_len

    @staticmethod
    def read_file(filename):
        """
        :param filename: csv file
        :return: [('/Users/cmm/数据集/BreaKHis_v1/histology_slides/breast/malignant/SOB/mucinous_carcinoma/SOB_M_MC_14-18842/100X/SOB_M_MC-14-18842-100-016.png', 'M', 'MC'),]
        """
        image_label_list = []
        with open(filename, 'r') as f:
            patients = csv.reader(f)
            for patient in patients:
                for root, dirs, files in os.walk(patient[0]):
                    for name in files:
                        if name.endswith('png'):
                            img_path = os.path.join(root, name)
                            labels = [patient[1], patient[2]]
                            image_label_list.append((img_path, labels))

        return image_label_list


def cifar10_dataset(train=True, data_path='./public', transform=None):
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),  # 转换成张量
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化
        ])
    return datasets.CIFAR10(root=data_path, train=train, download=True,
                            transform=transform)


def show(load_dataset, params):
    # print(BreakHisDataset.read_file(train_csv)[0])
    # print(len(BreakHisDataset.read_file(test_csv)))
    if 'transform' not in params.keys():
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        params['transform'] = transform
    print(params)
    train_data = load_dataset(**params)
    train_loader = DataLoader(train_data, batch_size=12, num_workers=0, shuffle=True)
    for i, (images, labels) in enumerate(train_loader):
        # show_image(image, labels[0][0])
        print("labels:", labels)
        if isinstance(labels, list):
            labels = labels[0]
        display_some_images(images, labels)
        break

# torch.manual_seed(42)
# data_path = 'C:/Users/CMM/Desktop/BreaKHis_v1/histology_slides/breast'
# show(cifar10_dataset, params={"train": True})
# show(BreakHisDataset, params={"data_path": "C:/Users/CMM/Desktop/BreaKHis_v1/histology_slides/breast",
#                               "train": True})
