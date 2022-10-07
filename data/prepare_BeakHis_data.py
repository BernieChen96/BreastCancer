# -*- coding: utf-8 -*-
# @Time    : 2022/10/2 09:25
# @Author  : CMM
# @Site    : 
# @File    : prepare_BeakHis_data.py
# @Software: PyCharm
import copy
import csv
import os
import random

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

# Description - Benign 良性肿瘤:
# adenosis 腺瘤 fibroadenoma 纤维肿瘤 phyllodes_tumor 叶状瘤 tubular_adenoma 管状瘤
benign_list = ['/benign/SOB/adenosis/', '/benign/SOB/fibroadenoma/', '/benign/SOB/phyllodes_tumor/',
               '/benign/SOB/tubular_adenoma/']
# Description - Malignant 恶性肿瘤:
# lobular_carcinoma 小叶癌 papillary_carcinoma 乳头状癌 ductal_carcinoma 导管癌 mucinous_carcinoma 粘液癌
malignant_list = ['/malignant/SOB/lobular_carcinoma/', '/malignant/SOB/papillary_carcinoma/',
                  '/malignant/SOB/ductal_carcinoma/', '/malignant/SOB/mucinous_carcinoma/']


def prepare(opt):
    # random seed
    random_state = opt.manual_seed
    random.seed(random_state)
    data_path = opt.data_path

    count = 0
    patient_list = []

    # Access benign categories patients
    for benign_type_dir in benign_list:
        p_dir_path = data_path + benign_type_dir
        for p_id in os.listdir(p_dir_path):
            if 'SOB' in p_id:
                patient_list.append(p_dir_path + p_id)
                count += 1

    # Access malignant categories patients
    for malignant_type_dir in malignant_list:
        p_dir_path = data_path + malignant_type_dir
        for p_id in os.listdir(p_dir_path):
            if 'SOB' in p_id:
                patient_list.append(p_dir_path + p_id)
                count += 1

    # Random shuffle the list and extract labels
    random.shuffle(patient_list)

    main_type_list = []
    sub_category_list = []

    # data: First column: Filepath, Second column: abstract category, Third column: concrete category
    with open(data_path + '/data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for patient_path in patient_list:
            main_type = patient_path.split('/')[-1].split('_')[1]
            main_type_list.append(main_type)
            sub_category = patient_path.split('/')[-1].split('_')[2]
            sub_category_list.append(sub_category)
            # print(patient_path, main_class, sub_class)
            writer.writerow([patient_path, main_type, sub_category])

    print(f"The number of patients is {count}")

    # 对 label 进行编码
    label_encode_type = LabelEncoder()
    label_encode_category = LabelEncoder()
    label_encode_type.fit(main_type_list)
    label_encode_category.fit(sub_category_list)
    encode_type = dict(zip(main_type_list, label_encode_type.transform(main_type_list)))
    encode_category = (dict(zip(sub_category_list, label_encode_category.transform(sub_category_list))))
    # 保存文件
    np.save(data_path + '/encode_type.npy', encode_type)
    np.save(data_path + '/encode_category.npy', encode_category)

    print(f"The encode of type is {encode_type}")
    print(f"The encode of category is {encode_category}")

    # 分层采样
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)

    for train_index, test_index in split.split(patient_list, sub_category_list):
        patient_list_train = [patient_list[index] for index in train_index]
        patient_list_test = [patient_list[index] for index in test_index]
        with open(data_path + '/train.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            for patient_path in patient_list_train:
                main_class = patient_path.split('/')[-1].split('_')[1]
                sub_class = patient_path.split('/')[-1].split('_')[2]
                writer.writerow([patient_path, main_class, sub_class])

        with open(data_path + '/test.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            for patient_path in patient_list_test:
                main_class = patient_path.split('/')[-1].split('_')[1]
                sub_class = patient_path.split('/')[-1].split('_')[2]
                writer.writerow([patient_path, main_class, sub_class])

    stat_dict = {}
    # stat
    for train_index, test_index in split.split(patient_list, sub_category_list):
        patient_list_train = [patient_list[index] for index in train_index]
        patient_list_test = [patient_list[index] for index in test_index]
        with open(data_path + '/stat.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['part', 'benign', 'malignant', 'DC', 'LC', 'MC', 'PC', 'PT', 'F', 'TA', 'A'])
            stat_dict['B'] = 0
            stat_dict['M'] = 0
            stat_dict['DC'] = 0
            stat_dict['LC'] = 0
            stat_dict['MC'] = 0
            stat_dict['PC'] = 0
            stat_dict['PT'] = 0
            stat_dict['F'] = 0
            stat_dict['TA'] = 0
            stat_dict['A'] = 0
            stat_dict_train = copy.deepcopy(stat_dict)
            stat_dict_test = copy.deepcopy(stat_dict)
            write_stat(patient_list, stat_dict, writer, 'all')
            write_stat(patient_list_train, stat_dict_train, writer, 'train')
            write_stat(patient_list_test, stat_dict_test, writer, 'test')


def write_stat(li, d, w, des):
    """
    统计数据状态
    :param li: 列表
    :param d: 字典
    :param w: writer
    :param des: 描述
    :return:
    """
    for patient in li:
        main_type = patient.split('/')[-1].split('_')[1]
        sub_category = patient.split('/')[-1].split('_')[2]
        d[main_type] += 1
        d[sub_category] += 1

    w.writerow([des, d['B'], d['M'], d['DC'], d['LC'], d['MC'],
                d['PC'], d['PT'], d['F'], d['TA'], d['A']])
