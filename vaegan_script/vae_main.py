from __future__ import print_function
import os
import numpy as np
import argparse
from datetime import datetime
import time
import seaborn
import pandas
import progressbar

import torch
import torch.nn as nn
import torch.utils.data
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable
from torchvision.utils import make_grid

from PIL import Image
import matplotlib.pyplot as plt

from cell_dataset import make_dataset
from cell_net import VaeGan
from utils import RollingMeasure


def main():
    parser = argparse.ArgumentParser(description='Capture cell features')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N')
    parser.add_argument('--epochs', type=int, default=100, metavar='N')
    parser.add_argument('----no_cuda', action='store_true', default=False)
    parser.add_argument('--datasets', '-d', default='./traindata1')
    parser.add_argument('--val', '-v', default='./testdata1')
    parser.add_argument('--out', '-o', default="./result_cell_image/result_{}".format(datetime.now().strftime("%Y%m%d")))
    parser.add_argument('--seed', type=int, default=1, metavar='S')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N')
    parser.add_argument('--z_size', default=256, type=int)
    parser.add_argument('--recon_level', default=3, action='store', type=int)
    parser.add_argument('--slurm', default=False, action='store', type=bool, dest="slurm")
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if not os.path.exists(args.out):
        os.mkdir(args.out)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.cuda else "cpu")
    print(device)

    files = os.listdir(args.datasets)
    cell_list = [f for f in files if os.path.isdir(os.path.join(args.dataset, f))]

    f = open(args.out + '/memo.txt', 'w')
    f.write(datetime.now().strftime("%Y/%m/%d  %H:%M:%S") + "\n" + "device : {}".format(device) + "\n" + "epoch : {}".format(args.epochs) + "\n" + "batch size : {}".format(args.batch_size) + "\n" + "latent dimensions : 512\n" + "image size : {} x {}".format(256, 256) + "\n" + "cells : {}".format(cell_list) + "\n")
    f.close()

    net = VaeGan(z_size=args.z_szie, recon_level=args.recon_level).to(device)

    margin = 0.35
    equilibrium = 0.68

    enc = net.encoder.parameters()
    dec = net.decoder.parameters()
    dis = net.discriminator.parameters()

    def make_optimizer(model, lr=0.0003, betas=(0.9, 0.999), weight_decay=0.00001):
        optimizer = Adam(model.parameters(), lr=lr, betas=betas)
        lr_optimizer = ExponentialLR(optimizer, gamma=weight_decay)
        lr_optimizer.setup(model)
        return optimizer

    lr_enc = make_optimizer(enc)
    lr_dec = make_optimizer(dec)
    lr_dis = make_optimizer(dis)

    if not args.slurm:
        train_data = torch.utils.data.DataLoader(make_dataset(args.datasets), batch_size=args.batch_size, suffle=True, num_workers=4)
        test_data = torch.utils.data.DataLoader(make_dataset(args.val), batch_size=100, suffle=False, num_workers=1)
    else:
        train_data = torch.utils.data.DataLoader(make_dataset(args.datasets), batch_size=args.batch_size, suffle=True, num_worker=4)
        test_data = torch.utils.data.DataLoader(make_dataset(args.val), batch_size=100, suffle=False, num_worker=1)

    batch_number = len(train_data)
    step_index = 0
    widgets = [

        'Batch : ', progressbar.Counter(),
        '/', progressbar.FormatCustomText('total : {}s'.format(batch_number)),
        ' ', progressbar.Bar(marker="-", left='[', right=']'),
        ' ', progressbar.ETA(),
        ' ',
        progressbar.DynamicMessage('loss_nle'),
        ' ',
        progressbar.DynamicMessage('loss_encoder'),
        ' ',
        progressbar.DynamicMessage('loss_decoder'),
        ' ',
        progressbar.DynamicMessage('loss_discriminator'),
        ' ',
        progressbar.DynamicMessage('loss_mse_layer'),
        ' ',
        progressbar.DynamicMessage('loss_kld'),
        ' ',
        progressbar.DynamicMessage("epoch")
    ]

    for i in range(args.epoch):

        progress = progressbar.ProgressBar(min_value=0, max_value=batch_number, initial_value=0,
                                           widgets=widgets).start()
        # reset rolling average
        loss_nle_mean = RollingMeasure()
        loss_encoder_mean = RollingMeasure()
        loss_decoder_mean = RollingMeasure()
        loss_discriminator_mean = RollingMeasure()
        loss_reconstruction_layer_mean = RollingMeasure()
        loss_kld_mean = RollingMeasure()
        gan_gen_eq_mean = RollingMeasure()
        gan_dis_eq_mean = RollingMeasure()
        #print("LR:{}".format(lr_encoder.get_lr()))

        # for each batch
        for j, (data_batch, target_batch, _) in enumerate(train_data):

            net.train()
            data_target = Variable(target_batch, requires_grad=False).float().cuda()
            data_in = Variable(data_batch, requires_grad=False).float().cuda()

            # get output
            out, out_labels, out_layer, mus, var = net(data_in)
            # split so we can get different parts
            out_layer_predicted = out_layer[:len(out_layer) // 2]
            out_layer_original = out_layer[len(out_layer) // 2:]
            # TODO set a batch_len variable to get a clean code here
            out_labels_original = out_labels[:len(out_labels) // 2]
            out_labels_sampled = out_labels[-len(out_labels) // 2:]
            # loss, nothing special here
            nle_value, kl_value, mse_value, bce_dis_original_value, bce_dis_sampled_value,\
                bce_gen_original_value, bce_gen_sampled_value = VaeGan.loss(data_target, out, out_layer_original,
                                                                         out_layer_predicted, out_labels_original,
                                                                          out_labels_sampled, mus, var)
            # THIS IS THE MOST IMPORTANT PART OF THE CODE
            loss_encoder = torch.sum(kl_value) + torch.sum(mse_value)
            loss_discriminator = torch.sum(bce_dis_original_value) + torch.sum(bce_dis_sampled_value)
            loss_decoder = torch.sum(lambda_mse * mse_value) - (1.0 - lambda_mse) * loss_discriminator
            #loss_decoder = torch.sum(mse_lambda * mse_value) + (1.0-mse_lambda)*(torch.sum(bce_gen_sampled_value)+torch.sum(bce_gen_original_value))

            # register mean values of the losses for logging
            loss_nle_mean(torch.mean(nle_value).data.cpu().numpy()[0])
            loss_discriminator_mean((torch.mean(bce_dis_original_value) + torch.mean(bce_dis_sampled_value)).data.cpu().numpy()[0])
            loss_decoder_mean((torch.mean(lambda_mse * mse_value) - (1 - lambda_mse) * (torch.mean(bce_dis_original_value) + torch.mean(bce_dis_sampled_value))).data.cpu().numpy()[0])
            #loss_decoder_mean((torch.mean(mse_lambda * mse_value) + (1-mse_lambda)*(torch.mean(bce_gen_original_value) + torch.mean(bce_gen_sampled_value))).data.cpu().numpy()[0])
            loss_encoder_mean((torch.mean(kl_value) + torch.mean(mse_value)).data.cpu().numpy()[0])
            loss_reconstruction_layer_mean(torch.mean(mse_value).data.cpu().numpy()[0])
            loss_kld_mean(torch.mean(kl_value).data.cpu().numpy()[0])
            # selectively disable the decoder of the discriminator if they are unbalanced
            train_dis = True
            train_dec = True
            if torch.mean(bce_dis_original_value).data[0] < equilibrium-margin or torch.mean(bce_dis_sampled_value).data[0] < equilibrium-margin:
                train_dis = False
            if torch.mean(bce_dis_original_value).data[0] > equilibrium+margin or torch.mean(bce_dis_sampled_value).data[0] > equilibrium+margin:
                train_dec = False
            if train_dec is False and train_dis is False:
                train_dis = True
                train_dec = True

            #aggiungo log
            if train_dis:
                gan_dis_eq_mean(1.0)
            else:
                gan_dis_eq_mean(0.0)

            if train_dec:
                gan_gen_eq_mean(1.0)
            else:
                gan_gen_eq_mean(0.0)

            # BACKPROP
            # clean grads
            net.zero_grad()
            # encoder
            loss_encoder.backward(retain_graph=True)
            # someone likes to clamp the grad here
            #[p.grad.data.clamp_(-1,1) for p in net.encoder.parameters()
            # update parameters
            lr_enc.step()
            # clean others, so they are not afflicted by encoder loss
            net.zero_grad()
            #decoder
            if train_dec:
                loss_decoder.backward(retain_graph=True)
                #[p.grad.data.clamp_(-1,1) for p in net.decoder.parameters()]
                lr_dec.step()
                #clean the discriminator
                net.discriminator.zero_grad()
            #discriminator
            if train_dis:
                loss_discriminator.backward()
                #[p.grad.data.clamp_(-1,1) for p in net.discriminator.parameters()]
                lr_dis.step()

            # LOGGING
            if args.slurm:
                progress.update(progress.value + 1, loss_nle=loss_nle_mean.measure,
                               loss_encoder=loss_encoder_mean.measure,
                               loss_decoder=loss_decoder_mean.measure,
                               loss_discriminator=loss_discriminator_mean.measure,
                               loss_mse_layer=loss_reconstruction_layer_mean.measure,
                               loss_kld=loss_kld_mean.measure,
                               epoch=i + 1)

        # EPOCH END
        if args.slurm:
            progress.update(progress.value + 1, loss_nle=loss_nle_mean.measure,
                            loss_encoder=loss_encoder_mean.measure,
                            loss_decoder=loss_decoder_mean.measure,
                            loss_discriminator=loss_discriminator_mean.measure,
                            loss_mse_layer=loss_reconstruction_layer_mean.measure,
                            loss_kld=loss_kld_mean.measure,
                            epoch=i + 1)
        lr_enc.step()
        lr_dec.step()
        lr_dis.step()
        decay_margin, decay_equilibrium, decay_mse = 1, 1, 1
        margin *=decay_margin
        equilibrium *=decay_equilibrium
        #margin non puo essere piu alto di equilibrium
        if margin > equilibrium:
            equilibrium = margin
        lambda_mse *=decay_mse
        if lambda_mse > 1:
            lambda_mse=1
        progress.finish()

        for j, (data_batch, target_batch, _) in enumerate(test_data):
            net.eval()

            data_in = Variable(data_batch, requires_grad=False).float().cuda()
            data_target = Variable(target_batch, requires_grad=False).float().cuda()
            out = net(data_in)
            out = out.data.cpu()
            out = (out + 1) / 2
            out = make_grid(out, nrow=8)

            out = net(None, 100)
            out = out.data.cpu()
            out = (out + 1) / 2
            out = make_grid(out, nrow=8)

            out = data_target.data.cpu()
            out = (out + 1) / 2
            out = make_grid(out, nrow=8)
            break

        step_index += 1
    exit(0)
