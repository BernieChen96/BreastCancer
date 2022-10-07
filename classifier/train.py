# -*- coding: utf-8 -*-
# @Time    : 2022/10/6 15:34
# @Author  : CMM
# @File    : train.py
# @Software: PyCharm
import os
import random
import time

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from classifier.model import Net



def set_seed(random_state):
    random.seed(random_state)
    # 为GPU设置种子
    torch.cuda.manual_seed(random_state)
    # 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.manual_seed(random_state)
    np.random.seed(random_state)


def init_dataloader(load_dataset, csv, data_path, preprocess, repeat=1, batch_size=16, num_workers=0, shuffle=True):
    dataset = load_dataset(csv, data_path, repeat=repeat, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    return dataloader


# 编码
# encode_type = np.load(root + '/encode_type.npy', allow_pickle='TRUE').item()
# decode_type = dict(zip(encode_type.values(), encode_type.keys()))
# encode_category = np.load(root + '/encode_category.npy', allow_pickle='TRUE').item()
# decode_category = dict(zip(encode_category.values(), encode_category.keys()))


class Trainer:

    def __init__(self, opt):
        # set seed
        set_seed(opt.manual_seed)
        # init model
        self.model = Net(opt).to(opt.device)
        # init dataloader
        train_csv = opt.data_path + '/train.csv'
        test_csv = opt.data_path + '/test.csv'
        train_loader = init_dataloader(opt.load_dataset, train_csv, opt.data_path, opt.weights.transforms())
        test_loader = init_dataloader(opt.load_dataset, test_csv, opt.data_path, opt.weights.transforms(),
                                      shuffle=False)
        self.dataloaders = {'train': train_loader, 'val': test_loader}
        self.dataset_sizes = {'train': train_loader.dataset.__len__(), 'val': test_loader.dataset.__len__()}
        # init criterion & optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    def train_step(self, best_acc, opt):
        # 每个epoch都有一个训练和验证阶段，每个epoch都会历遍一次整个训练集
        for phase in ['train', 'val']:
            if phase == 'train':
                # 如果模型中有BN层(Batch Normalization, 批量归一化）和Dropout，需要在训练时添加model.train()，在测试时添加model.eval()。
                # 其中model.train()是保证BN层用每一批数据的均值和方差，而model.eval()是保证BN用全部训练数据的均值和方差；
                # 而对于Dropout，model.train()是随机取一部分网络连接来训练更新参数，而model.eval()是利用到了所有网络连接。
                # 训练过程中使用model.train()，作用是启用batch normalization和drop out。
                self.model.train()  # set model to training mode
            else:
                # 测试过程中使用model.eval()，这时神经网络会沿用batch normalization的值，并不使用drop out。
                self.model.eval()  # set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0.0

            # 迭代数据
            for inputs, labels in tqdm(self.dataloaders[phase]):
                # .to(device): copy一份到device所指定的GPU上去，之后的运算都在GPU上进行。
                labels = labels[opt.label_class]
                inputs = inputs.to(opt.device)
                labels = labels.to(opt.device)

                # 梯度参数清零
                # 因为如果不清零，那么使用的这个grad就得同上一个mini-batch有关，这不是我们需要的结果。
                self.optimizer.zero_grad()

                # 与with torch.no_grad() 相似，参数是一个bool类型的值，决定了梯度计算启动（True）或禁用（False）。
                # 训练阶段启动自动求导，验证阶段关掉梯度计算，节省eval的时间；预测阶段并不反向求导，只有前向计算，求导及梯度更新只出现在训练阶段
                # 只进行inference时，model.eval()是必须使用的，否则会影响结果准确性。 而torch.no_grad()并不是强制的，只影响运行效率。
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # 前向计算
                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)  # 1表示取每行的最大值；preds 表示只返回最大值所在位置
                    # 计算loss
                    loss = self.criterion(outputs, labels.long())

                    # 后向计算 + 仅在训练阶段进行优化
                    if phase == 'train':
                        # optimizer更新参数空间需要基于反向梯度
                        loss.backward()
                        # optimizer.step()通常用在每个mini-batch之中，更新模型参数
                        # mini-batch训练模式是假定每一个训练集就只有mini-batch这样大，一次训练更新一次参数空间
                        self.optimizer.step()
                # 在每个epoch内累积统计running_loss
                running_loss += loss.item() * inputs.size(0)  # .size(0)取行数
                # .data 取出本体tensor数据，舍弃了grad，grad_fn等额外反向图计算过程需保存的额外信息。
                running_corrects += torch.sum(preds == labels.data)
                # print(phase, running_loss, running_corrects)

            # 计算损失的平均值
            epoch_loss = running_loss / self.dataset_sizes[phase]
            epoch_acc = running_corrects.double() / self.dataset_sizes[phase]
            # 精度，召回率和F1度量
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # 跟踪最佳性能的模型（从验证准确率方面），并在训练结束时返回性能最好的模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                # 保存模型
                torch.save(self.model.state_dict(), opt.checkpoint)
        return best_acc

    def train(self, opt):
        start = time.perf_counter()
        # Record the best accuracy
        best_acc = 0.0
        # Load the best checkpoint state
        if os.path.exists(opt.checkpoint):
            state_dict = torch.load(opt.checkpoint)
            self.model.load_state_dict(state_dict)
        for epoch in range(opt.epochs):
            print('Epoch {}/{}'.format(epoch, opt.epochs - 1))
            print('-' * 10)
            best_acc = self.train_step(best_acc, opt)
        end = time.perf_counter()
        time_elapsed = end - start
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:.4f}'.format(best_acc))
