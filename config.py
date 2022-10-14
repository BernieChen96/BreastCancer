# -*- coding: utf-8 -*-
# @Time    : 2022/10/3 00:47
# @Author  : CMM
# @Site    : 
# @File    : config.py
# @Software: PyCharm
import datetime
import os
import argparse
import random
import numpy as np
import torch
from torchvision import transforms
from torchvision.models import DenseNet201_Weights, EfficientNet_B0_Weights
from data.load_data import BreakHisDataset, cifar10_dataset
from torch.utils.data import DataLoader, random_split

from ddpm.ddpm import script_utils

root = os.path.abspath(os.path.dirname(__file__))


# ------ Basic configuration ------
def get_basic_argument():
    parser = argparse.ArgumentParser("basic configuration")
    parser.add_argument('--root', default=root, help='project root path')
    parser.add_argument('--dataset', default='BreakHis',
                        help='Select dataset.')
    parser.add_argument('--manual_seed', default=42, type=int, help='manual random seed')
    return parser


def get_dataset_path(dataset):
    data_path = root
    if dataset == 'BreakHis':
        data_path = root + '/public/BreaKHis_v1/histology_slides/breast'
    elif dataset == 'cifar10':
        data_path = root + '/public'
    else:
        print(f'{dataset} does not exists')
        exit(0)
    return data_path


def set_seed(random_state):
    random.seed(random_state)
    # 为GPU设置种子
    torch.cuda.manual_seed(random_state)
    # 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.manual_seed(random_state)
    np.random.seed(random_state)


# ------ classifier configuration ------
def get_classifier_argument():
    parser = get_basic_argument()
    parser.add_argument('--model', default='densenet201',
                        help='Select classification model, densenet201, efficientnet-b0')
    parser.add_argument('--c', default='type', help='Select class to classify. ')
    parser.add_argument('--lr', default=0.0005, type=float, help='learning rate')
    parser.add_argument('--b1', default=0.5, type=float, help='learning rate')
    parser.add_argument('--b2', default=0.999, type=float, help='learning rate')
    parser.add_argument('--epochs', default=20, type=int, help='epochs')
    parser.add_argument('--proportion', default=1, type=float, help='Training data proportion')
    return parser


def post_classifier_config(opt):
    opt.data_path = get_dataset_path(opt.dataset)
    opt.root = opt.root + '/classifier'
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if opt.model == 'densenet201':
        opt.weights = DenseNet201_Weights.DEFAULT
    elif opt.model == 'efficientnet-b0':
        opt.weights = EfficientNet_B0_Weights.DEFAULT
    else:
        print(f'{opt.model} does not exist.')
        exit(0)
    set_seed(opt.manual_seed)
    print("Random Seed: ", opt.manual_seed)
    checkpoint_dir = opt.root + f'/checkpoint/{opt.model}'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    plots_dir = opt.root + f'/plots/{opt.model}'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    if opt.dataset == 'BreakHis':
        train_dataset = BreakHisDataset(opt.data_path,
                                        train=True,
                                        repeat=1,
                                        transform=opt.weights.transforms())
        opt.train_loader = DataLoader(train_dataset,
                                      batch_size=16,
                                      num_workers=0,
                                      shuffle=True)
        opt.test_loader = DataLoader(BreakHisDataset(opt.data_path,
                                                     train=False,
                                                     repeat=1,
                                                     transform=opt.weights.transforms()),
                                     batch_size=16,
                                     num_workers=0,
                                     shuffle=False)
        opt.checkpoint = checkpoint_dir + f'/{opt.dataset}_{opt.c}_best.pth'
        opt.plots = plots_dir + f'/{opt.dataset}_{opt.c}_result.npy'
        if opt.c == 'type':
            opt.n_classes = 2
            opt.label_class = 0
            opt.decode = train_dataset.decode_type
        elif opt.c == 'category':
            opt.n_classes = 8
            opt.label_class = 1
            opt.decode = train_dataset.decode_category
    elif opt.dataset == 'cifar10':
        opt.n_classes = 10
        opt.label_class = None
        train_dataset = cifar10_dataset(train=True,
                                        data_path=opt.data_path,
                                        transform=opt.weights.transforms())
        train_length = train_dataset.__len__()
        part_train_length = int(opt.proportion * train_length)
        train_dataset = random_split(train_dataset, [part_train_length, train_length - part_train_length])[0]
        print("The training dataset length is:", train_dataset.__len__())
        opt.train_loader = DataLoader(train_dataset,
                                      batch_size=16,
                                      num_workers=0,
                                      shuffle=True)
        opt.test_loader = DataLoader(cifar10_dataset(data_path=opt.data_path,
                                                     train=False,
                                                     transform=opt.weights.transforms()
                                                     ),
                                     batch_size=16,
                                     num_workers=0,
                                     shuffle=False)
        opt.checkpoint = checkpoint_dir + f'/{opt.dataset}_{opt.proportion}_best.pth'
        opt.plots = plots_dir + f'/{opt.dataset}_{opt.proportion}_result.npy'
    else:
        print(f'{opt.dataset} does not exist.')
        exit(0)


# ------ score configuration ------
def get_metrics_argument():
    parser = get_basic_argument()
    parser.add_argument('--model', default='densenet201',
                        help='Select classification model, densenet201, efficientnet-b0')
    parser.add_argument('--c', default='type', help='Select class to classify. ')
    return parser


def post_metrics_config(opt):
    opt.data_path = get_dataset_path(opt.dataset)
    opt.report_dir = opt.root + f'/report/{opt.model}'
    if opt.model == 'densenet201' or opt.model == 'efficientnet-b0':
        opt.model_type = 'classifier'
        opt.checkpoint = opt.root + f'/classifier/checkpoint/{opt.model}/{opt.dataset}_{opt.c}_best.pth'
        opt.result_path = opt.report_dir + f'/{opt.dataset}_{opt.c}_result.npy'
        opt.prediction_path = opt.report_dir + f'/{opt.dataset}_{opt.c}_prediction.png'
        opt.acc_report_path = opt.report_dir + f'/{opt.dataset}_{opt.c}_acc_report.csv'
        opt.confusion_metrix_path = opt.report_dir + f'/{opt.dataset}_{opt.c}_confusion_metrix.png'
        opt.roc_path = opt.report_dir + f'/{opt.dataset}_{opt.c}_roc.png'
    elif opt.model == '':
        pass
    else:
        print(f'{opt.model} does not exist.')

    if not os.path.exists(opt.checkpoint):
        print('Checkpoint does not exists')
        exit(0)
    if not os.path.exists(opt.report_dir):
        os.makedirs(opt.report_dir)


# ------ generative adversarial network configuration ------

def get_gan_argument():
    parser = get_basic_argument()
    parser.add_argument('--n_z', default=100, type=int,
                        help='Hidden space dimension')
    parser.add_argument('--n_c', type=int, required=True,
                        help='Number of categories')
    parser.add_argument('--img_size', type=int, required=True, help='Image size')
    parser.add_argument('--channels', type=int, default=3, help='Image channel')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--b1', type=float, default=0.5, help='Beta1')
    parser.add_argument('--b2', type=float, default=0.999, help='Beta2')
    parser.add_argument('--epochs', type=int, default=200, help='Epochs')
    parser.add_argument('--sample_interval', type=int, default=400, help='Generate sample interval')
    parser.add_argument('--proportion', default=0.1, type=float, help='Training data proportion')
    parser.add_argument('--model', default='acgan', help='Select generation model')
    return parser


def post_gan_config(opt):
    opt.data_path = get_dataset_path(opt.dataset)
    opt.root = opt.root + '/gan'
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(opt.manual_seed)
    print("Random Seed: ", opt.manual_seed)
    opt.checkpoint_dir = opt.root + f'/checkpoint/{opt.model}_{opt.dataset}_{opt.proportion}'
    if not os.path.exists(opt.checkpoint_dir):
        os.makedirs(opt.checkpoint_dir)
    opt.sample_dir = opt.root + f'/sample/{opt.model}_{opt.dataset}_{opt.proportion}'
    if not os.path.exists(opt.sample_dir):
        os.makedirs(opt.sample_dir)
    plots_dir = opt.root + f'/plots/{opt.model}'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    if opt.dataset == 'cifar10':
        opt.n_classes = 10
        transform = transforms.Compose(
            [transforms.Resize((opt.img_size, opt.img_size)), transforms.ToTensor(),
             transforms.Normalize([0.5], [0.5])]
        )
        dataset = cifar10_dataset(train=True,
                                  data_path=opt.data_path,
                                  transform=transform)
        length = dataset.__len__()
        part_length = int(opt.proportion * length)
        train_dataset = random_split(dataset, [part_length, length - part_length])[0]
        print("The training dataset length is:", train_dataset.__len__())
        opt.dataloader = DataLoader(train_dataset,
                                    batch_size=16,
                                    num_workers=0,
                                    shuffle=True)


# ------ ddpm configuration ------
def get_ddpm_argument():
    parser = get_basic_argument()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    run_name = datetime.datetime.now().strftime("ddpm-%Y-%m-%d-%H-%M")
    defaults = dict(
        learning_rate=2e-4,
        batch_size=128,
        iterations=800000,

        log_to_wandb=True,
        log_rate=1000,
        checkpoint_rate=1000,
        log_dir=root + "/ddpm/ddpm_logs",
        project_name='ddpm',
        run_name=run_name,

        model_checkpoint=None,
        optim_checkpoint=None,

        schedule_low=1e-4,
        schedule_high=0.02,

        device=device,
    )
    defaults.update(script_utils.diffusion_defaults())
    defaults['use_labels'] = True
    script_utils.add_dict_to_argparser(parser, defaults)
    # parser.add_argument('--use_labels', type=bool, default=False, help='Whether to use label')
    return parser


def post_ddpm_config(opt):
    set_seed(opt.manual_seed)
    print("Random Seed: ", opt.manual_seed)
    opt.data_path = get_dataset_path(opt.dataset)
    if not os.path.exists(opt.log_dir):
        os.makedirs(opt.log_dir)


# ------ data preparation configuration ------
def get_data_preparation_argument():
    parser = get_basic_argument()

    return parser
