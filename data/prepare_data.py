# -*- coding: utf-8 -*-
# @Time    : 2022/10/2 09:25
# @Author  : CMM
# @Site    : 
# @File    : prepare_data.py
# @Software: PyCharm
import copy
import csv
import os
import random

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

import config

# random seed
random_state = config.random_state
random.seed(random_state)

root = config.data_root

# Description - Benign 良性肿瘤:
# adenosis 腺瘤 fibroadenoma 纤维肿瘤 phyllodes_tumor 叶状瘤 tubular_adenoma 管状瘤
benign_list = ['/benign/SOB/adenosis/', '/benign/SOB/fibroadenoma/', '/benign/SOB/phyllodes_tumor/',
               '/benign/SOB/tubular_adenoma/']
# Description - Malignant 恶性肿瘤:
# lobular_carcinoma 小叶癌 papillary_carcinoma 乳头状癌 ductal_carcinoma 导管癌 mucinous_carcinoma 粘液癌
malignant_list = ['/malignant/SOB/lobular_carcinoma/', '/malignant/SOB/papillary_carcinoma/',
                  '/malignant/SOB/ductal_carcinoma/', '/malignant/SOB/mucinous_carcinoma/']

count = 0
patient_list = []

# Access benign categories patients
for benign_type_dir in benign_list:
    p_dir_path = root + benign_type_dir
    for p_id in os.listdir(p_dir_path):
        if 'SOB' in p_id:
            patient_list.append(p_dir_path + p_id)
            count += 1

# Access malignant categories patients
for malignant_type_dir in malignant_list:
    p_dir_path = root + malignant_type_dir
    for p_id in os.listdir(p_dir_path):
        if 'SOB' in p_id:
            patient_list.append(p_dir_path + p_id)
            count += 1

# Random shuffle the list and extract labels
random.shuffle(patient_list)

main_type_list = []
sub_category_list = []

# data: First column: Filepath, Second column: abstract category, Third column: concrete category
with open(root + '/data.csv', 'w') as f:
    writer = csv.writer(f)
    for patient_path in patient_list:
        main_class = patient_path.split('/')[-1].split('_')[1]
        main_type_list.append(main_class)
        sub_class = patient_path.split('/')[-1].split('_')[2]
        sub_category_list.append(sub_class)
        print(patient_path, main_class, sub_class)
        writer.writerow([patient_path, main_class, sub_class])

print(f"The number of patients is {count}")

writer = csv.writer(f)
le_type = LabelEncoder()
le_category = LabelEncoder()
le_type.fit(main_type_list)
le_category.fit(sub_category_list)
encode = dict(zip(main_type_list, le_type.transform(main_type_list)))
encode.update(dict(zip(sub_category_list, le_category.transform(sub_category_list))))
# 保存文件
np.save(root + '/encode.npy', encode)

stat_dict = {}
stat_dict_train = {}
stat_dict_test = {}
# 分层采样
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)

for train_index, test_index in split.split(patient_list, sub_category_list):
    patient_list_train = [patient_list[index] for index in train_index]
    patient_list_test = [patient_list[index] for index in test_index]
    with open(root + '/train.csv', 'w') as f:
        writer = csv.writer(f)
        for patient_path in patient_list_train:
            main_class = patient_path.split('/')[-1].split('_')[1]
            sub_class = patient_path.split('/')[-1].split('_')[2]
            writer.writerow([patient_path, main_class, sub_class])

    with open(root + '/test.csv', 'w') as f:
        writer = csv.writer(f)
        for patient_path in patient_list_test:
            main_class = patient_path.split('/')[-1].split('_')[1]
            sub_class = patient_path.split('/')[-1].split('_')[2]
            writer.writerow([patient_path, main_class, sub_class])


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


# stat
for train_index, test_index in split.split(patient_list, sub_category_list):
    patient_list_train = [patient_list[index] for index in train_index]
    patient_list_test = [patient_list[index] for index in test_index]
    with open(root + '/stat.csv', 'w') as f:
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
