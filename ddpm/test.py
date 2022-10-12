# -*- coding: utf-8 -*-
# @Time    : 2022/10/10 21:46
# @Author  : CMM
# @File    : test.py
# @Software: PyCharm
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8)
).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size=128,
    timesteps=1000,  # number of steps
    sampling_timesteps=250,
    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    loss_type='l1'  # L1 or L2
).cuda()

trainer = Trainer(
    diffusion,
    'C:/Users/CMM/Desktop/BreaKHis_v1/histology_slides/breast/benign/SOB/adenosis/SOB_B_A_14-22549AB/40X',
    train_batch_size=32,
    train_lr=8e-5,
    train_num_steps=1000,  # total training steps
    gradient_accumulate_every=2,  # gradient accumulation steps
    ema_decay=0.995,  # exponential moving average decay
    amp=True  # turn on mixed precision
)

trainer.train()
