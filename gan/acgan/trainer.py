# -*- coding: utf-8 -*-
# @Time    : 2022/8/28 22:41
# @Author  : CMM
# @Site    : 
# @File    : trainer.py
# @Software: PyCharm
import torch
from gan.acgan.model import Generator, Discriminator
import torchvision.utils as vutils


class Trainer:
    def __init__(self, opt):
        super(Trainer, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fixed_noise = None
        self.fixed_labels = None
        self.optimizer_G = None
        self.optimizer_D = None
        self.adversarial_loss = None
        self.auxiliary_loss = None
        self.net_D = None
        self.net_G = None
        self.real_label = None
        self.fake_label = None
        self.setup(opt)

    def setup(self, opt):
        # Initialize generator and discriminator
        self.net_G = Generator(opt).to(self.device)
        self.net_D = Discriminator(opt).to(self.device)
        # if config.NET_G != '':
        #     self.net_G.load_state_dict(torch.load(config.NET_G))
        # print(self.net_G)
        # if config.NET_D != '':
        #     self.net_D.load_state_dict(torch.load(config.NET_D))
        # print(self.net_D)

        # Loss Function
        self.adversarial_loss = torch.nn.BCELoss()
        self.auxiliary_loss = torch.nn.CrossEntropyLoss()

        # Initialize optimizer
        self.optimizer_G = torch.optim.Adam(self.net_G.parameters(), opt.lr, betas=(opt.b1, opt.b2))
        self.optimizer_D = torch.optim.Adam(self.net_D.parameters(), opt.lr, betas=(opt.b1, opt.b2))

        self.fixed_noise = torch.randn(opt.batch, opt.n_z, device=self.device)
        self.fixed_labels = torch.arange(0, opt.batch, device=self.device)
        for i in range(len(self.fixed_labels)):
            self.fixed_labels[i] = self.fixed_labels[i] % opt.n_c
        self.real_label = 1
        self.fake_label = 0

        # # 检查模型
        # self.summary_graph(self.net_G, (self.fixed_noise, self.fixed_labels))
        # fake = self.net_G(self.fixed_noise, self.fixed_labels)
        # self.summary_graph(self.net_D, (fake.detach(), self.fixed_labels))
        # # 查看数据
        # self.summary_embedding(self.dataset, self.classes)

    def train(self, opt):
        for epoch in range(opt.epochs):
            for i, (imgs, labels) in enumerate(opt.dataloader):
                labels = labels.long()
                # Adversarial ground truths
                real_imgs = imgs.to(self.device)
                real_labels = labels.to(self.device)
                batch_size = real_imgs.shape[0]
                valid_label = torch.full((batch_size, 1), self.real_label,
                                         dtype=real_imgs.dtype, device=self.device)
                fake_label = torch.full((batch_size, 1), self.fake_label, dtype=real_imgs.dtype, device=self.device)
                ############################
                # (1) Update G network: maximize log(D(G(z)))
                ###########################
                self.optimizer_G.zero_grad()
                noise = torch.randn(batch_size, opt.n_z, device=self.device)
                gen_labels = torch.randint(0, opt.n_c, (batch_size,), device=self.device)
                fake_image = self.net_G(noise, gen_labels)
                validity, pred_label = self.net_D(fake_image)
                g_loss = 0.5 * (
                        self.adversarial_loss(validity, valid_label) + self.auxiliary_loss(pred_label, gen_labels))
                g_loss.backward()
                self.optimizer_G.step()
                ############################
                # (2) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                self.optimizer_D.zero_grad()
                # train with real
                real_pred, real_aux = self.net_D(real_imgs)
                d_real_loss = (self.adversarial_loss(real_pred, valid_label) + self.auxiliary_loss(real_aux,
                                                                                                   real_labels)) / 2

                # train with fake
                fake_pred, fake_aux = self.net_D(fake_image.detach())
                d_fake_loss = (self.adversarial_loss(fake_pred, fake_label) + self.auxiliary_loss(fake_aux,
                                                                                                  gen_labels)) / 2

                # Total discriminator loss
                d_loss = (d_real_loss + d_fake_loss) / 2
                d_loss.backward()
                self.optimizer_D.step()

                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f'
                      % (epoch, opt.epochs, i, len(opt.dataloader),
                         d_loss.item(), g_loss.item()))

                if i % 100 == 0:
                    fake = self.net_G(self.fixed_noise, self.fixed_labels)
                    vutils.save_image(real_imgs,
                                      opt.sample_dir + '/real_samples.png',
                                      normalize=True)
                    vutils.save_image(fake.detach(),
                                      opt.sample_dir + '/fake_samples_epoch_%03d.png' % epoch,
                                      normalize=True)
            # do checkpointing
            if epoch % 5 == 0:
                torch.save(self.net_G.state_dict(), opt.checkpoint_dir+'/netG_epoch_%d.pth' % epoch)
                torch.save(self.net_D.state_dict(), opt.checkpoint_dir+'/netD_epoch_%d.pth' % epoch)
