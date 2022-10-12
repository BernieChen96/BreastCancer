# -*- coding: utf-8 -*-
# @Time    : 2022/10/2 15:40
# @Author  : CMM
# @Site    : 
# @File    : image_processing.py
# @Software: PyCharm
import matplotlib.pyplot as plt
import numpy as np


def show_image(image, title, is_norm=True):
    """
    调用matplotlib显示RGB图片
    :param is_norm: 是否进行过标准化
    :param title: 图像标题
    :param image: 图像的数据
    :return:
    """
    if image.ndim > 3:
        image = image[0]
    if is_norm:
        image = image / 2 + 0.5
    npimg = image.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    plt.imshow(npimg)
    plt.axis('off')  # 关掉坐标轴为 off
    plt.title(title)  # 图像题目
    plt.show()


def display_some_images(images, labels, is_norm=True):
    fig = plt.figure(figsize=(15, 15))
    columns = 4
    rows = 3
    cmap = None
    if images.shape[1] == 1:
        cmap = 'gray'
    for i in range(0, columns * rows):
        ax = fig.add_subplot(rows, columns, i + 1)
        ax.title.set_text(labels[i].numpy())
        if is_norm:
            image = images[i] / 2 + 0.5
        npimg = image.numpy()
        npimg = np.transpose(npimg, (1, 2, 0))
        plt.imshow(npimg, interpolation='nearest', cmap=cmap)
    plt.tight_layout()
    plt.show()
