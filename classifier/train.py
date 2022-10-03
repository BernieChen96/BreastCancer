# -*- coding: utf-8 -*-
# @Time    : 2022/10/3 17:29
# @Author  : CMM
# @File    : train.py
# @Software: PyCharm
# -*- coding: utf-8 -*-
# @Time    : 2022/10/2 17:44
# @Author  : CMM
# @Site    :
# @File    : train.py
# @Software: PyCharm
import copy
import os.path
import time
from tqdm import tqdm
import numpy as np
import torch
import warnings
from classifier.model import Net
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.models import DenseNet201_Weights

import config
from data.load_data import BreakHisDataset, train_csv, test_csv, decode_type, decode_category
from score import performance_indicators as pi

warnings.filterwarnings("ignore")

# img = read_image("/Users/cmm/Projects/PycharmProjects/projects/breastcancer/cat.jpeg")
weights = DenseNet201_Weights.DEFAULT
# # Step 2: Initialize the inference transforms
preprocess = weights.transforms()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_data = BreakHisDataset(train_csv, repeat=1, transform=preprocess)
train_loader = DataLoader(train_data, batch_size=16, num_workers=0, shuffle=True)
test_data = BreakHisDataset(test_csv, repeat=1, transform=preprocess)
test_loader = DataLoader(test_data, batch_size=16, num_workers=0, shuffle=True)
dataloaders = {'train': train_loader, 'val': test_loader}
dataset_sizes = {'train': train_data.__len__(), 'val': test_data.__len__()}
print(dataset_sizes)

num_classes = config.classifier_num_classes
classifier_class = config.classifier_class
if classifier_class == 0:
    decode = decode_type
elif classifier_class == 1:
    decode = decode_category


# 所有训练样本都已输入到模型中，称为一个epoch
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    # copy.copy(xxx)浅拷贝只拷贝最外层的数值和指针，不会去拷贝最外层“指针”所指向的内层的东西
    # copy.deepcopy(xxx)深拷贝则会拷贝全部层的东西
    # torch.nn.Module模块中的state_dict 保存模型中的weight权值和bias偏置值
    best_acc = 0.0

    if os.path.exists(config.classifier_checkpoint):
        state_dict = torch.load(config.classifier_checkpoint)
        model.load_state_dict(state_dict)
        best_model_wts = state_dict
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
            for inputs, labels in tqdm(dataloaders[phase]):
                # .to(device): copy一份到device所指定的GPU上去，之后的运算都在GPU上进行。
                labels = labels[classifier_class]
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
                    _, preds = torch.max(outputs, 1)  # 1表示取每行的最大值；preds 表示只返回最大值所在位置
                    # 计算loss
                    loss = criterion(outputs, labels.long())

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
                # print(phase, running_loss, running_corrects)

            # 计算损失的平均值
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            # 精度，召回率和F1度量

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # 跟踪最佳性能的模型（从验证准确率方面），并在训练结束时返回性能最好的模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())  # 深度拷贝
                # 保存模型
                torch.save(model.state_dict(), config.classifier_checkpoint)
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:.4f}'.format(best_acc))

    # 加载最佳模型权重
    # load_state_dict将预训练的参数权重加载到新的模型中
    model.load_state_dict(best_model_wts)
    return model


def visualize_model(model, num_images=8):
    # 默认情况下model.train()，model.training为True;
    # 若model.eval()，model.training为False
    was_training = model.training
    # 使用model.eval()时，它指示模型不需要学习任何新的内容，并且模型用于测试。
    # model.eval()等价于model.train(mode=False)
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():  # 验证阶段关掉梯度计算
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            # .to(device): copy到device所指定的GPU上去
            labels = labels[classifier_class]
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):  # 等价于.size(0)
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')

                ax.set_title('predicted: {}, origin: {}'.format(decode[preds[j].cpu().item()],
                                                                decode[labels[j].cpu().item()]))
                # .cpu是把数据转移到cpu上, .data是读取Variable中的tensor本体
                # 如果tensor是放在GPU上，先得用.cpu()把它传到cpu上
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    plt.tight_layout()
                    plt.show()
                    model.train(mode=was_training)  # False，测试模式
                    return
            plt.tight_layout()
            plt.show()
        model.train(mode=was_training)


def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)


def fine_tuning():
    classifier_model = Net(num_classes=num_classes, w=weights)
    # 把 model.parameters() 移动到GPU上面去
    classifier_model = classifier_model.to(device)
    # 交叉熵损失函数CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(classifier_model.parameters(), lr=0.0005, betas=(0.5, 0.999))

    # torch.optim.lr_scheduler.StepLR 根据epoch训练次数来调整学习率
    # 更新策略：每过step_size个epoch，做一次更新；gamma是更新lr的乘法因子
    # 每7个epochs衰减LR通过设置gamma=0.1
    # exp_lr_scheduler 更新epoch
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    classifier_model = train_model(classifier_model, criterion, optimizer, exp_lr_scheduler, num_epochs=20)
    return classifier_model


def load_model():
    classifier_model = Net(num_classes=num_classes, w=weights)
    # 把 model.parameters() 移动到GPU上面去
    classifier_model = classifier_model.to(device)
    if os.path.exists(config.classifier_checkpoint):
        state_dict = torch.load(config.classifier_checkpoint)
        classifier_model.load_state_dict(state_dict)
    return classifier_model


def classification_report(model):
    model.eval()
    running_corrects = 0.0
    y_true = []
    y_pred = []
    targets = []
    for i in range(len(decode)):
        targets.append(decode[i])
    with torch.no_grad():  # 验证阶段关掉梯度计算
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            # .to(device): copy到device所指定的GPU上去
            labels = labels[classifier_class]
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.data.tolist())
            y_pred.extend(preds.tolist())
            running_corrects += torch.sum(preds == labels.data)
    acc = running_corrects.double() / dataset_sizes['val']
    print("验证集上的准确率：", acc)
    pi.report(y_true, y_pred, target_names=targets)
    print(pi.save_classification_report(y_true, y_pred, target_names=targets, save_path=config.classifier_acc_report))


if __name__ == '__main__':
    if config.classifier_mode == 'train':
        model = fine_tuning()
        visualize_model(model)
    elif config.classifier_mode == 'report':
        model = load_model()
        classification_report(model)
