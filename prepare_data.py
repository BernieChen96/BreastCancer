# -*- coding: utf-8 -*-
# @Time    : 2022/10/7 15:15
# @Author  : CMM
# @File    : prepare_data.py
# @Software: PyCharm
import config
from data.prepare_BeakHis_data import prepare

if __name__ == '__main__':
    parser = config.get_data_preparation_argument()
    opt = parser.parse_args()
    print('Perform data preparation task.')
    opt.data_path = config.get_dataset_path(opt.dataset)
    prepare(opt)
