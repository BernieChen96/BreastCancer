# -*- coding: utf-8 -*-
# @Time    : 2022/10/2 13:32
# @Author  : CMM
# @Site    : 
# @File    : load_data.py
# @Software: PyCharm
import csv
import os
import random
import torch
import config
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from data.image_processing import display_some_images

root = config.data_root

train_csv = root + '/train.csv'
test_csv = root + '/test.csv'

random_state = config.random_state
random.seed(random_state)
# 为GPU设置种子
torch.cuda.manual_seed(random_state)
# 为CPU设置种子用于生成随机数，以使得结果是确定的
torch.manual_seed(random_state)
np.random.seed(random_state)

# 编码
encode = np.load(root + '/encode.npy', allow_pickle='TRUE').item()


class BreakHisDataset(Dataset):  # 需要继承data.Dataset
    def __init__(self, filename, repeat=1, transform=None):
        """
        :param filename: 数据文件TXT：格式：imge_name.jpg label1_id labe2_id
        :param resize_height 为None时，不进行缩放
        :param resize_width  为None时，不进行缩放，
        PS：当参数resize_height或resize_width其中一个为None时，可实现等比例缩放
        :param repeat: 所有样本数据重复次数，默认循环一次，当repeat为None时，表示无限循环<sys.maxsize
        """
        # TODO
        # 1. Initialize file path or list of file names.
        self.image_label_list = self.read_file(filename)
        self.len = len(self.image_label_list)
        self.repeat = repeat

        self.transform = transform

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
                            labels = [encode[patient[1]], encode[patient[2]]]
                            image_label_list.append((img_path, labels))

        return image_label_list


def show():
    # print(BreakHisDataset.read_file(train_csv)[0])
    # print(len(BreakHisDataset.read_file(test_csv)))
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_data = BreakHisDataset(train_csv, repeat=None, transform=transform)
    train_loader = DataLoader(train_data, batch_size=12, num_workers=0, shuffle=True)
    for i, (images, labels) in enumerate(train_loader):
        # show_image(image, labels[0][0])
        print(labels)
        display_some_images(images, labels)
        return


show()
