# -*- coding: utf-8 -*-
# @Time    : 2022/10/7 16:31
# @Author  : CMM
# @File    : classify.py
# @Software: PyCharm
import config
from classifier import functions
from classifier.train import Trainer

if __name__ == '__main__':
    parser = config.get_classifier_argument()
    opt = parser.parse_args()
    config.post_classifier_config(opt)
    trainer = Trainer(opt)
    trainer.train(opt)
