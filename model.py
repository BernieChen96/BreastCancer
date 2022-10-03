# -*- coding: utf-8 -*-
# @Time    : 2022/10/2 17:44
# @Author  : CMM
# @Site    : 
# @File    : model.py
# @Software: PyCharm
import copy
import time
import torch
import warnings

from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.models import densenet201, DenseNet201_Weights

from data.load_data import BreakHisDataset, train_csv, test_csv

warnings.filterwarnings("ignore")

# img = read_image("/Users/cmm/Projects/PycharmProjects/projects/breastcancer/cat.jpeg")
weights = DenseNet201_Weights.DEFAULT
# # Step 2: Initialize the inference transforms
preprocess = weights.transforms()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_data = BreakHisDataset(train_csv, repeat=1, transform=preprocess)
train_loader = DataLoader(train_data, batch_size=16, num_workers=0, shuffle=True)
test_data = BreakHisDataset(test_csv, repeat=1, transform=preprocess)
test_loader = DataLoader(train_data, batch_size=16, num_workers=0)
dataloaders = {'train': train_loader, 'val': test_loader}
dataset_sizes = {'train': train_data.__len__(), 'val': test_data.__len__()}
print(dataset_sizes)


# 所有训练样本都已输入到模型中，称为一个epoch
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    # copy.copy(xxx)浅拷贝只拷贝最外层的数值和指针，不会去拷贝最外层“指针”所指向的内层的东西
    # copy.deepcopy(xxx)深拷贝则会拷贝全部层的东西
    # torch.nn.Module模块中的state_dict 保存模型中的weight权值和bias偏置值
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 每个epoch都有一个训练和验证阶段，每个epoch都会历遍一次整个训练集
        for phase in ['train', 'val']:
            if phase == 'train':
                # scheduler.step()通常用在epoch里面，更新优化器的学习率lr
                # UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()；
                # 解决：warnings.filterwarnings("ignore")
                scheduler.step()
                # 如果模型中有BN层(Batch Normalization, 批量归一化）和Dropout，需要在训练时添加model.train()，在测试时添加model.eval()。
                # 其中model.train()是保证BN层用每一批数据的均值和方差，而model.eval()是保证BN用全部训练数据的均值和方差；
                # 而对于Dropout，model.train()是随机取一部分网络连接来训练更新参数，而model.eval()是利用到了所有网络连接。
                # 训练过程中使用model.train()，作用是启用batch normalization和drop out。
                model.train()  # set model to training mode
            else:
                # 测试过程中使用model.eval()，这时神经网络会沿用batch normalization的值，并不使用drop out。
                model.eval()  # set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0

            # 迭代数据
            for inputs, labels in dataloaders[phase]:
                # .to(device): copy一份到device所指定的GPU上去，之后的运算都在GPU上进行。
                labels = labels[0]
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 梯度参数清零
                # 因为如果不清零，那么使用的这个grad就得同上一个mini-batch有关，这不是我们需要的结果。
                optimizer.zero_grad()

                # 与with torch.no_grad() 相似，参数是一个bool类型的值，决定了梯度计算启动（True）或禁用（False）。
                # 训练阶段启动自动求导，验证阶段关掉梯度计算，节省eval的时间；预测阶段并不反向求导，只有前向计算，求导及梯度更新只出现在训练阶段
                # 只进行inference时，model.eval()是必须使用的，否则会影响结果准确性。 而torch.no_grad()并不是强制的，只影响运行效率。
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # 前向计算
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)  # 1表示取每行的最大值；_,表示只返回最大值所在位置
                    # 计算loss
                    loss = criterion(outputs, labels)

                    # 后向计算 + 仅在训练阶段进行优化
                    if phase == 'train':
                        # optimizer更新参数空间需要基于反向梯度
                        loss.backward()
                        # optimizer.step()通常用在每个mini-batch之中，更新模型参数
                        # mini-batch训练模式是假定每一个训练集就只有mini-batch这样大，一次训练更新一次参数空间
                        optimizer.step()

                # 在每个epoch内累积统计running_loss
                running_loss += loss.item() * inputs.size(0)  # .size(0)取行数
                # .data 取出本体tensor数据，舍弃了grad，grad_fn等额外反向图计算过程需保存的额外信息。
                running_corrects += torch.sum(preds == labels.data)
                print("training:", running_loss, running_corrects)

            # 计算损失的平均值
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # 跟踪最佳性能的模型（从验证准确率方面），并在训练结束时返回性能最好的模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())  # 深度拷贝

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:.4f}'.format(best_acc))

    # 加载最佳模型权重
    # load_state_dict将预训练的参数权重加载到新的模型中
    model.load_state_dict(best_model_wts)
    return model


class ClassificationModel(nn.Module):
    def __init__(self):
        super(ClassificationModel, self).__init__()
        weights = DenseNet201_Weights.DEFAULT
        model = densenet201(weights=weights)


def fine_tuning():
    # Step 1: Initialize model with the best available weights
    model = densenet201(weights=weights)
    # 内层与预先训练的模型保持一致，只有最后的层被更改以适应我们的类数量
    # 修改预训练模型的参数
    # 提取classifier（全连接）层中固定的参数
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, 2)
    # print(model)
    # 把 model.parameters() 移动到GPU上面去
    model = model.to(device)
    # 交叉熵损失函数CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()
    # SGD指stochastic gradient descent，即随机梯度下降。是梯度下降的batch版本。
    # 缺点是，其更新方向完全依赖于当前的batch
    # 引入momentum动量法，模拟物体运动时的惯性，即更新的时候在一定程度上保留之前更新的方向
    # 参考：https://blog.csdn.net/tsyccnh/article/details/76270707
    # optimizer_ft 更新mini-batch
    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # torch.optim.lr_scheduler.StepLR 根据epoch训练次数来调整学习率
    # 更新策略：每过step_size个epoch，做一次更新；gamma是更新lr的乘法因子
    # 每7个epochs衰减LR通过设置gamma=0.1
    # exp_lr_scheduler 更新epoch
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)


fine_tuning()
