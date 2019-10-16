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

from cell_net import VaeGan
from cell_dataset import make_dataset
from utils import RollingMeasure


def main():
    parser = argparse.ArgumentParser(description='Capture cell features')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N')
    parser.add_argument('--epochs', type=int, default=100, metavar='N')
    parser.add_argument('----no_cuda', action='store_true', default=False)
    parser.add_argument('--datasets', '-d', default='./dataset')
    parser.add_argument('--out', '-o',
                        default="./result_cell_image/result_{}".format(datetime.now().strftime("%Y%m%d")))
    parser.add_argument('--seed', type=int, default=1, metavar='S')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N')
    parser.add_argument('--z_size', default=256, type=int)
    parser.add_argument('--recon_level', default=3, action='store', type=int)
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.cuda else "cpu")
    print(device)
    cell_list = [x for x in os.listdir(args.datasets)]

    f = open(args.out + '/memo.txt', 'w')
    f.write(datetime.now().strftime("%Y/%m/%d %H:%M:%S") + "\n"
            "device : {}\nepochs : {}\nbatch size : {}\n"
            "latent dimensions : {}\nimage size : 256 x 256\n"
            "cells : {}".format(device, args.epochs, args.batch_size, args.z_size, cell_list))
    f.close()

    model = VaeGan(z_size=args.z_size, recon_level=args.recon_level).to(device)

    margin = 0.35
    equilibrium = 0.68

    enc = model.encoder.parameters()
    dec = model.decoder.parameters()
    dis = model.discriminator.parameters()

    def make_optimizer(model, lr=0.0003, betas=(0.9, 0.99), weight_decay=0.00001):
        optimizer = Adam(model, lr=lr, betas=betas)
        lr_optimizer = ExponentialLR(optimizer, gamma=weight_decay)
        return lr_optimizer

    lr_enc = make_optimizer(enc)
    lr_dec = make_optimizer(dec)
    lr_dis = make_optimizer(dis)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    traindata = make_dataset(args.datasets, list=cell_list, data_range=(1, 100))
    testdata = make_dataset(args.datasets, list=cell_list, data_range=(100, 140))

    trainloader = torch.utils.data.DataLoader(traindata, batch_size=args.batch_size, shuffle=True, **kwargs)
    testloader = torch.utils.data.DataLoader(testdata, batch_size=1, shuffle=False)

    batch_number = len(trainloader)
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

    enc_loss = []
    dec_loss = []
    dis_loss = []

    for i in range(args.epochs):

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
        # print("LR:{}".format(lr_encoder.get_lr()))

        # for each batch
        for j, (data_batch, target_batch) in enumerate(trainloader):

            train_batch = len(data_batch)
            model.train()
            data_target = data_batch.to(device)
            data_in = data_batch.to(device)

            # get output
            out, out_labels, out_layer, mus, var = model(data_in)
            # split so we can get different parts
            out_layer_predicted = out_layer[:train_batch]
            out_layer_original = out_layer[train_batch:-train_batch]
            out_layer_sampled = out_layer[-train_batch:]
            # labels
            out_labels_predicted = out_labels[:train_batch]
            out_labels_original = out_labels[train_batch:-train_batch]
            out_labels_sampled = out_labels[-train_batch:]
            # loss, nothing special here
            nle_value, kl_value, mse_value_1, mse_value_2, bce_dis_original_value, bce_dis_sampled_value, \
            bce_dis_predicted_value, bce_gen_sampled_value, bce_gen_predicted_value = VaeGan.loss(data_target, out, out_layer_original, out_layer_predicted, out_layer_sampled, out_labels_original, out_labels_predicted, out_labels_sampled, mus, var)
            # THIS IS THE MOST IMPORTANT PART OF THE CODE
            loss_encoder = torch.sum(kl_value) + torch.sum(mse_value_1) + torch.sum(mse_value_2)
            loss_discriminator = torch.sum(bce_dis_original_value) + torch.sum(bce_dis_sampled_value) + torch.sum(bce_dis_predicted_value)
            loss_decoder = torch.sum(bce_gen_sampled_value) + torch.sum(bce_gen_predicted_value)
            loss_decoder = torch.sum(lambda_mse * mse_value_1) - (1.0 - lambda_mse) * loss_decoder
            # loss_decoder = torch.sum(mse_lambda * mse_value) + (1.0-mse_lambda)*(torch.sum(bce_gen_sampled_value)+torch.sum(bce_gen_original_value))

            enc_loss.append(loss_encoder)
            dec_loss.append(loss_decoder)
            dis_loss.append(loss_discriminator)

            # register mean values of the losses for logging
            loss_nle_mean(torch.mean(nle_value).data.cpu().numpy()[0])
            loss_discriminator_mean(
                (torch.mean(bce_dis_original_value) + torch.mean(bce_dis_sampled_value)).data.cpu().numpy()[0])
            loss_decoder_mean((torch.mean(lambda_mse * mse_value_1 / 2) + torch.mean(lambda_mse * mse_value_2 / 2) + (
                        1 - lambda_mse) * (torch.mean(bce_gen_predicted_value) + torch.mean(
                bce_gen_sampled_value))).data.cpu().numpy()[0])
            loss_encoder_mean(
                (torch.mean(kl_value) + torch.mean(mse_value_1) + torch.mean(mse_value_2)).data.cpu().numpy()[0])
            loss_reconstruction_layer_mean((torch.mean(mse_value_1) + torch.mean(mse_value_2)).data.cpu().numpy()[0])
            loss_kld_mean(torch.mean(kl_value).data.cpu().numpy()[0])
            # selectively disable the decoder of the discriminator if they are unbalanced
            train_dis = True
            train_dec = True
            if torch.mean(bce_dis_original_value).data[0] < equilibrium - margin or \
                    torch.mean(bce_dis_sampled_value).data[0] < equilibrium - margin:
                train_dis = False
            if torch.mean(bce_dis_original_value).data[0] > equilibrium + margin or \
                    torch.mean(bce_dis_sampled_value).data[0] > equilibrium + margin:
                train_dec = False
            if train_dec is False and train_dis is False:
                train_dis = True
                train_dec = True

            # aggiungo log
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
            model.zero_grad()
            # encoder
            loss_encoder.backward()
            # someone likes to clamp the grad here
            # [p.grad.data.clamp_(-1,1) for p in net.encoder.parameters()
            # update parameters
            lr_enc.step(args.epochs)
            # clean others, so they are not afflicted by encoder loss
            model.zero_grad()
            # decoder
            if train_dec:
                loss_decoder.backward()
                # [p.grad.data.clamp_(-1,1) for p in net.decoder.parameters()]
                lr_dec.step(args.epochs)
                # clean the discriminator
                model.discriminator.zero_grad()
            # discriminator
            if train_dis:
                loss_discriminator.backward()
                # [p.grad.data.clamp_(-1,1) for p in net.discriminator.parameters()]
                lr_dis.step(args.epochs)

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
        lr_enc.step(args.epochs)
        lr_dec.step(args.epochs)
        lr_dis.step(args.epochs)
        decay_margin, decay_equilibrium, decay_mse = 1, 1, 1
        margin *= decay_margin
        equilibrium *= decay_equilibrium
        # margin non puo essere piu alto di equilibrium
        if margin > equilibrium:
            equilibrium = margin
        lambda_mse *= decay_mse
        if lambda_mse > 1:
            lambda_mse = 1
        progress.finish()

        for j, (data_batch, target_batch) in enumerate(testloader):
            model.eval()

            data_in = data_batch.to(device)
            data_target = data_batch.to(device)
            out = model(data_in)
            out = out.data.cpu()
            out = (out + 1) / 2
            out = make_grid(out, nrow=8)

            out = model(None, 100)
            out = out.data.cpu()
            out = (out + 1) / 2
            out = make_grid(out, nrow=8)

            out = data_target.data.cpu()
            out = (out + 1) / 2
            out = make_grid(out, nrow=8)
            break

        step_index += 1
    exit(0)

    np.save(args.out + '/enc_loss.npy', np.array(enc_loss))
    np.save(args.out + '/dec_loss.npy', np.array(dec_loss))
    np.save(args.out + '/dis_loss.npy', np.array(dis_loss))
    torch.save(model.state_dict(), args.out + '/vae.pth')


if __name__ == '__main__':
    main()
