# -*- coding: utf-8 -*-
# @Time    : 2022/10/14 10:07
# @Author  : CMM
# @File    : gan_images.py
# @Software: PyCharm
import config
from gan.acgan.trainer import Trainer

if __name__ == '__main__':
    parser = config.get_gan_argument()
    opt = parser.parse_args()
    config.post_gan_config(opt)
    trainer = Trainer(opt)
    trainer.train(opt)
