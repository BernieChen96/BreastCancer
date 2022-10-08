# -*- coding: utf-8 -*-
# @Time    : 2022/10/7 20:55
# @Author  : CMM
# @File    : run_metrics.py
# @Software: PyCharm
import math
import os
import random
import numpy as np
import torch
import config
from matplotlib import pyplot as plt
from torch import nn

from classifier import functions
from classifier.model import Net
from classifier.train import init_dataloader
from score import performance_indicators as pi


def set_seed(random_state):
    random.seed(random_state)
    # 为GPU设置种子
    torch.cuda.manual_seed(random_state)
    # 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.manual_seed(random_state)
    np.random.seed(random_state)


def visualize_model(opt, model, dataloader, decode):
    # 默认情况下model.train()，model.training为True;
    # 使用model.eval()时，它指示模型不需要学习任何新的内容，并且模型用于测试。
    # model.eval()等价于model.train(mode=False)
    model.eval()
    plt.figure(figsize=(8, 8))
    for i, (inputs, labels) in enumerate(dataloader):
        # .to(device): copy到device所指定的GPU上去
        labels = labels[opt.label_class]
        inputs = inputs.to(opt.device)
        labels = labels.to(opt.device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        batch = inputs.size()[0]
        col = math.floor(math.sqrt(batch))
        row = int(batch / col) if (batch % col) == 0 else int(math.floor(batch / col) + 1)
        for j in range(0, batch):  # 等价于.size(0)
            ax = plt.subplot(row, col, j + 1)
            ax.axis('off')
            ax.set_title('predicted: {}, origin: {}'.format(decode[preds[j].cpu().item()],
                                                            decode[labels[j].cpu().item()]))
            # .cpu是把数据转移到cpu上, .data是读取Variable中的tensor本体
            # 如果tensor是放在GPU上，先得用.cpu()把它传到cpu上
            imshow(inputs.cpu().data[j])
        plt.tight_layout()
        plt.savefig(opt.prediction_path, bbox_inches='tight')
        plt.show()
        return


def load_model(opt):
    if opt.model_type == 'classifier':
        config.post_classifier_config(opt)
        model = Net(opt).to(opt.device)
        # Load the best checkpoint state
        if os.path.exists(opt.checkpoint):
            print('Load checkpoint')
            state_dict = torch.load(opt.checkpoint)
            model.load_state_dict(state_dict)
    else:
        model = None
    return model


def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)


def classification_result(opt, model, dataloader, decode):
    model.eval()
    running_corrects = 0.0
    y_true = []
    y_pred = []
    y_pred_prob = []
    target = []
    for i in range(len(decode)):
        target.append(decode[i])
    for i, (inputs, labels) in enumerate(dataloader):
        # .to(device): copy到device所指定的GPU上去
        labels = labels[opt.label_class]
        inputs = inputs.to(opt.device)
        labels = labels.to(opt.device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        pred_prob = nn.functional.softmax(outputs, dim=1)
        y_true.extend(labels.data.tolist())
        y_pred.extend(preds.tolist())
        y_pred_prob.extend(pred_prob.tolist())
        running_corrects += torch.sum(preds == labels.data)
    acc = running_corrects.double() / dataloader.dataset.__len__()
    print("验证集上的准确率：", acc.item())

    result_dict = {'y_true': y_true, 'y_pred': y_pred, 'y_pred_prob': y_pred_prob, 'target': target}
    np.save(opt.result_path, result_dict)


def classifier_report(opt):
    result_dict = np.load(opt.result_path, allow_pickle=True).item()
    y_true = result_dict['y_true']
    y_pred = result_dict['y_pred']
    y_pred_prob = result_dict['y_pred_prob']
    target = result_dict['target']
    # acc report
    print(pi.save_classification_report(y_true, y_pred, target_names=target, save_path=opt.acc_report_path))
    # confusion matrix
    pi.plot_confusion_matrix(y_true, y_pred, title='Confusion Metrix for Breast Cancer', classes=target,
                             save_path=opt.confusion_metrix_path)
    # auc
    if len(target) == 2:
        pi.plot_roc(y_true, y_pred, save_path=opt.roc_path)
    else:
        pi.plot_roc_mutil(y_true, y_pred_prob, target, save_path=opt.roc_path)


def main(opt):
    if opt.model == 'densenet201' or opt.model == 'efficientnet-b0':
        test_csv = opt.data_path + '/test.csv'
        model = load_model(opt)
        test_dataloader = init_dataloader(opt.load_dataset, test_csv, opt.data_path, opt.weights.transforms(),
                                          shuffle=True)
        if opt.label_class == 0:
            decode = test_dataloader.dataset.decode_type
        else:
            decode = test_dataloader.dataset.decode_category
        visualize_model(opt, model, test_dataloader, decode)
        if not os.path.exists(opt.result_path):
            classification_result(opt, model, test_dataloader, decode)
        classifier_report(opt)
    elif opt.model == '':
        pass


if __name__ == '__main__':
    parser = config.get_metrics_argument()
    opt = parser.parse_args()
    config.post_metrics_config(opt)
    main(opt)
