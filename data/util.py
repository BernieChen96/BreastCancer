# -*- coding: utf-8 -*-
# @Time    : 2022/10/16 11:15
# @Author  : CMM
# @File    : util.py
# @Software: PyCharm
import os.path

import torchvision
import numpy as np
import imageio  # 引入imageio包
import config

train_data = torchvision.datasets.CelebA(root=config.root + '/public',
                                         transform=torchvision.transforms.ToTensor(), download=True)


# test_data = torchvision.datasets.CIFAR10(root=config.root + '/public', train=False,
#                                          transform=torchvision.transforms.ToTensor(),
#                                          download=True)


# 解压 返回解压后的字典
def unpickle(file):
    import pickle as pk
    fo = open(file, 'rb')
    dict = pk.load(fo, encoding='iso-8859-1')
    fo.close()
    return dict

# begin unpickle
# root_dir = config.root + '/public/cifar-10-batches-py'
#
# if not os.path.exists(root_dir + '/train/'):
#     os.mkdir(root_dir + '/train/')
#     os.mkdir(root_dir + '/test/')
# # 生成训练集图片
# for j in range(1, 6):
#     dataName = root_dir + "/data_batch_" + str(j)  # 读取当前目录下的data_batch1~5文件。
#     Xtr = unpickle(dataName)
#     print(dataName + " is loading...")
#
#     for i in range(0, 10000):
#         img = np.reshape(Xtr['data'][i], (3, 32, 32))  # Xtr['data']为图片二进制数据
#         img = img.transpose(1, 2, 0)  # 读取image
#         picName = root_dir + '/train/' + str(Xtr['labels'][i]) + '_' + str(i + (j - 1) * 10000) + '.jpg'
#         imageio.imsave(picName, img)  # 使用的imageio的imsave类
#     print(dataName + " loaded.")
#
# print("test_batch is loading...")
#
# # 生成测试集图片
# testXtr = unpickle(root_dir + "/test_batch")
# for i in range(0, 10000):
#     img = np.reshape(testXtr['data'][i], (3, 32, 32))
#     img = img.transpose(1, 2, 0)
#     picName = root_dir + '/test/' + str(testXtr['labels'][i]) + '_' + str(i) + '.jpg'
#     imageio.imsave(picName, img)
# print("test_batch loaded.")
