# -*- coding: utf-8 -*-
# @Time    : 2022/10/3 14:52
# @Author  : CMM
# @File    : model.py
# @Software: PyCharm
import torch
from torch import nn
from torchvision.models import densenet201, DenseNet201_Weights


class Net(nn.Module):

    def __init__(self, num_classes=2, w=DenseNet201_Weights.DEFAULT):
        super(Net, self).__init__()
        model = densenet201(weights=w)
        in_features = model.classifier.in_features
        # print(model)
        self.backbone = nn.Sequential(*list(model.children())[0])
        # print(self.backbone)
        self.tail = nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout(0.5)
        )
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        out = self.tail(features)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
