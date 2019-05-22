import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import sys


class CycleGANModel(BaseModel):
    def name(self):
        return 'CycleGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize
        self.with_disc = opt.with_disc
        self.cycle_gan = opt.cycle_gan
        self.dis_loss_thr = opt.dis_loss_thr
        self.dis_thr = opt.dis_thr

        self.loss_G_A = -1
        self.loss_G_B = -1
        self.loss_cycle_A = -1
        self.loss_cycle_B = -1
        self.loss_D_A = -1
        self.loss_D_B = -1
        self.loss_D_RA = -1
        self.loss_D_RB = -1

        if self.with_disc:
            self.loss_G_RB = -1
            self.loss_G_RA = -1

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.vgg = networks.Vgg19().cuda()

        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        if self.with_disc and self.isTrain:
            self.discriminatorA = networks.PerceptualDiscriminator(gpu_ids=self.gpu_ids, start_lay=opt.start_lay_dis, norm_layer=networks.get_norm_layer(norm_type=opt.norm)).cuda()
            self.discriminatorB = networks.PerceptualDiscriminator(gpu_ids=self.gpu_ids, start_lay=opt.start_lay_dis, norm_layer=networks.get_norm_layer(norm_type=opt.norm)).cuda()
            networks.init_weights(self.discriminatorA, init_type=opt.init_type)
            networks.init_weights(self.discriminatorB, init_type=opt.init_type)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A, 'G_A', which_epoch)
            self.load_network(self.netG_B, 'G_B', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', which_epoch)
                self.load_network(self.netD_B, 'D_B', which_epoch)
                if self.with_disc:
                    self.load_network(self.discriminatorA, 'D_AP', which_epoch)
                    self.load_network(self.discriminatorB, 'D_BP', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionVGG = networks.VGGLoss(self.gpu_ids, opt.start_lay)
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=0.5*opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=0.5*opt.lr, betas=(opt.beta1, 0.999))
            if self.with_disc:
                self.optimizer_D_RA = torch.optim.Adam(self.discriminatorA.parameters(), lr=0.5*opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_D_RB = torch.optim.Adam(self.discriminatorB.parameters(), lr=0.5*opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D_A)
            self.optimizers.append(self.optimizer_D_B)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_A)
        networks.print_network(self.netG_B)
        if self.isTrain:
            networks.print_network(self.netD_A)
            networks.print_network(self.netD_B)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], async=True)
            input_B = input_B.cuda(self.gpu_ids[0], async=True)
        self.input_A = input_A
        self.input_B = input_B
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)

    def test(self):
        real_A = Variable(self.input_A, volatile=True)
        fake_B = self.netG_A(real_A)
        self.rec_A = self.netG_B(fake_B).data
        self.fake_B = fake_B.data

        real_B = Variable(self.input_B, volatile=True)
        fake_A = self.netG_B(real_B)
        self.rec_B = self.netG_A(fake_A).data
        self.fake_A = fake_A.data

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
        # self.train_D_A = True
        # if loss_D_A.data[0] < self.dis_loss_thr and self.dis_thr:
        #     self.train_D_A = False
        if self.loss_D_A == -1:
            self.loss_D_A = loss_D_A.data[0]
        else:
            self.loss_D_A = 0.6 * self.loss_D_A + 0.4 * loss_D_A.data[0]

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
        # self.train_D_B = True
        # if loss_D_B.data[0] < self.dis_loss_thr and self.dis_thr:
        #     self.train_D_B = False
        if self.loss_D_B == -1:
            self.loss_D_B = loss_D_B.data[0]
        else:
            self.loss_D_B = 0.6 * self.loss_D_B + 0.4 * loss_D_B.data[0]

    def backward_D_RA(self):
        pred_real = self.discriminatorA(self.A_vgg[1].detach(), self.A_vgg[2].detach(), self.A_vgg[3].detach(), self.A_vgg[4].detach())
        pred_fake = self.discriminatorA(self.A_fake_vgg[1].detach(), self.A_fake_vgg[2].detach(), self.A_fake_vgg[3].detach(), self.A_fake_vgg[4].detach())

        loss_D_real = self.criterionGAN(pred_real, True)
        loss_D_fake = self.criterionGAN(pred_fake, False)
        loss_D_R = (loss_D_real + loss_D_fake) * 0.5
        loss_D_R.backward(retain_graph=True)
        if self.loss_D_RA == -1:
            self.loss_D_RA = loss_D_R.data[0]
        else:
            self.loss_D_RA = 0.6 * self.loss_D_RA + 0.4 * loss_D_R.data[0]

    def backward_D_RB(self):
        pred_real = self.discriminatorB(self.B_vgg[1].detach(), self.B_vgg[2].detach(), self.B_vgg[3].detach(), self.B_vgg[4].detach())
        pred_fake = self.discriminatorB(self.B_fake_vgg[1].detach(), self.B_fake_vgg[2].detach(), self.B_fake_vgg[3].detach(), self.B_fake_vgg[4].detach())

        loss_D_real = self.criterionGAN(pred_real, True)
        loss_D_fake = self.criterionGAN(pred_fake, False)
        loss_D_R = (loss_D_real + loss_D_fake) * 0.5
        loss_D_R.backward(retain_graph=True)
        if self.loss_D_RB == -1:
            self.loss_D_RB = loss_D_R.data[0]
        else:
            self.loss_D_RB = 0.6 * self.loss_D_RB + 0.4 * loss_D_R.data[0]

    def backward_G(self):
        lambda_idt = self.opt.identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            idt_A = self.netG_A(self.real_B)
            loss_idt_A = self.criterionIdt(idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
            idt_B = self.netG_B(self.real_A)
            loss_idt_B = self.criterionIdt(idt_B, self.real_A) * lambda_A * lambda_idt

            self.idt_A = idt_A.data
            self.idt_B = idt_B.data
            self.loss_idt_A = loss_idt_A.data[0]
            self.loss_idt_B = loss_idt_B.data[0]
        else:
            loss_idt_A = 0
            loss_idt_B = 0
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        fake_B = self.netG_A(self.real_A)
        pred_fake = self.netD_A(fake_B)
        loss_G_A = self.criterionGAN(pred_fake, True)

        # GAN loss D_B(G_B(B))
        fake_A = self.netG_B(self.real_B)
        pred_fake = self.netD_B(fake_A)
        loss_G_B = self.criterionGAN(pred_fake, True)

        # Forward cycle loss
        # noise = Variable(torch.FloatTensor(fake_B.size()).uniform_(0.1, 1) * 0.02, requires_grad=False).cuda()
        rec_A = self.netG_B(fake_B)
        A_vgg, A_rec_vgg, A_fake_vgg = self.vgg(self.real_A), self.vgg(rec_A), self.vgg(fake_A)
        loss_cycle_A = self.criterionVGG(A_vgg, A_rec_vgg) * self.opt.lambda_feat
        loss_G_RA = 0
        if self.with_disc:
            pred_fake = self.discriminatorA(A_fake_vgg[1], A_fake_vgg[2], A_fake_vgg[3], A_fake_vgg[4])
            loss_G_RA = self.criterionGAN(pred_fake, True)
        if self.cycle_gan:
            pred_fake = self.netD_B(rec_A)
            loss_G_A += self.criterionGAN(pred_fake, True)

        # Backward cycle loss
        # noise = Variable(torch.FloatTensor(fake_B.size()).uniform_(0.1, 1) * 0.02, requires_grad=False).cuda()
        rec_B = self.netG_A(fake_A)
        B_vgg, B_rec_vgg, B_fake_vgg = self.vgg(self.real_B), self.vgg(rec_B), self.vgg(fake_B)
        loss_cycle_B = self.criterionVGG(B_vgg, B_rec_vgg) * self.opt.lambda_feat
        loss_G_RB = 0
        if self.with_disc:
            pred_fake = self.discriminatorB(B_fake_vgg[1], B_fake_vgg[2], B_fake_vgg[3], B_fake_vgg[4])
            loss_G_RB = self.criterionGAN(pred_fake, True)
        if self.cycle_gan:
            pred_fake = self.netG_A(rec_B)
            loss_G_B += self.criterionGAN(pred_fake, True)

        # tv_loss = REGULARIZATION * (torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:]))
        #                             + torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])))

        # combined loss
        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B + loss_G_RA + loss_G_RB
        loss_G.backward()

        self.fake_B = fake_B.data
        self.fake_A = fake_A.data
        self.rec_A = rec_A.data
        self.rec_B = rec_B.data
        self.A_vgg = A_vgg
        self.B_vgg = B_vgg
        self.A_fake_vgg = A_fake_vgg
        self.B_fake_vgg = B_fake_vgg

        if self.loss_G_A == -1:
            self.loss_G_A = loss_G_A.data[0]
            self.loss_G_B = loss_G_B.data[0]
            self.loss_cycle_A = loss_cycle_A.data[0]
            self.loss_cycle_B = loss_cycle_B.data[0]
            if self.with_disc:
                self.loss_G_RB = loss_G_RB.data[0]
                self.loss_G_RA = loss_G_RA.data[0]
        else:
            self.loss_G_A = 0.6 * self.loss_G_A + 0.4 * loss_G_A.data[0]
            self.loss_G_B = 0.6 * self.loss_G_B + 0.4 * loss_G_B.data[0]
            self.loss_cycle_A = 0.6 * self.loss_cycle_A + 0.4 * loss_cycle_A.data[0]
            self.loss_cycle_B = 0.6 * self.loss_cycle_B + 0.4 * loss_cycle_B.data[0]
            if self.with_disc:
                self.loss_G_RB = 0.6 * self.loss_G_RB + 0.4 * loss_G_RB.data[0]
                self.loss_G_RA = 0.6 * self.loss_G_RA + 0.4 * loss_G_RA.data[0]

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_A
        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        # if self.train_D_A:
        self.optimizer_D_A.step()
        # D_B
        self.optimizer_D_B.zero_grad()
        self.backward_D_B()
        # if self.train_D_A:
        self.optimizer_D_B.step()
        if self.with_disc:
            # D_R
            self.optimizer_D_RA.zero_grad()
            self.backward_D_RA()
            self.optimizer_D_RA.step()
            # D_R
            self.optimizer_D_RB.zero_grad()
            self.backward_D_RB()
            self.optimizer_D_RB.step()

    def get_current_errors(self):
        ret_errors = OrderedDict([('D_A', self.loss_D_A), ('G_A', self.loss_G_A), ('Cyc_A', self.loss_cycle_A),
                                 ('D_B', self.loss_D_B), ('G_B', self.loss_G_B), ('Cyc_B',  self.loss_cycle_B)])
        if self.with_disc:
            ret_errors['D_RA'] = self.loss_D_RA
            ret_errors['D_RB'] = self.loss_D_RB
            ret_errors['G_RA'] = self.loss_G_RA
            ret_errors['G_RB'] = self.loss_G_RB
        if self.opt.identity > 0.0:
            ret_errors['idt_A'] = self.loss_idt_A
            ret_errors['idt_B'] = self.loss_idt_B
        return ret_errors

    def get_current_visuals(self):
        real_A = util.tensor2im(self.input_A)
        fake_B = util.tensor2im(self.fake_B)
        rec_A = util.tensor2im(self.rec_A)
        real_B = util.tensor2im(self.input_B)
        fake_A = util.tensor2im(self.fake_A)
        rec_B = util.tensor2im(self.rec_B)
        ret_visuals = OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A),
                                   ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B)])
        if self.opt.isTrain and self.opt.identity > 0.0:
            ret_visuals['idt_A'] = util.tensor2im(self.idt_A)
            ret_visuals['idt_B'] = util.tensor2im(self.idt_B)
        return ret_visuals

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
        self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
        self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)
        if self.with_disc:
           self.save_network(self.discriminatorA, 'D_AP', label, self.gpu_ids)
           self.save_network(self.discriminatorB, 'D_BP', label, self.gpu_ids)