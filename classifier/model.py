# -*- coding: utf-8 -*-
# @Time    : 2022/10/7 17:03
# @Author  : CMM
# @File    : model.py
# @Software: PyCharm
from torch import nn
from torchvision.models import densenet201, efficientnet_b0, regnet_y_3_2gf


class Net(nn.Module):

    def __init__(self, opt):
        super(Net, self).__init__()
        if opt.model == 'densenet201':
            # 224*224*3
            self.model = densenet201(weights=opt.weights)
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(in_features, opt.n_classes)
            )
        elif opt.model == 'efficientnet-b0':
            # 224*224*3
            self.model = efficientnet_b0(weights=opt.weights)
            self.model.classifier = nn.Sequential(
                nn.Dropout(0.2, inplace=True),
                nn.Linear(1280, opt.n_classes)
            )

    def forward(self, x):
        out = self.model(x)
        return out
