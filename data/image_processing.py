# -*- coding: utf-8 -*-
# @Time    : 2022/10/2 15:40
# @Author  : CMM
# @Site    : 
# @File    : image_processing.py
# @Software: PyCharm
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms


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


def dataset_detail(load_dataset, params):
    if 'transform' not in params.keys():
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        params['transform'] = transform
    dataset = load_dataset(**params)
    loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True)
    print(f"Dataset length is {dataset.__len__()}")
    images, labels = next(iter(loader))
    channel = images.shape[1]
    height = images.shape[2]
    width = images.shape[3]
    print(f"Image channel is {images.shape[1]}")
    print(f"Image height is {images.shape[2]}")
    print(f"Image width is {images.shape[3]}")

    return channel, height, width
