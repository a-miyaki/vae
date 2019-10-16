import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding, stride):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, padding=padding,
                              stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(num_features=out_ch, momentum=0.9)

    def forward(self, ten, out=False, t=False):
        if out:
            ten = self.conv(ten)
            ten_out = ten
            ten = self.bn(ten)
            ten = F.relu(ten, False)
            return ten, ten_out
        else:
            ten = self.conv(ten)
            ten = self.bn(ten)
            ten = F.relu(ten, True)
            return ten


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding, stride):
        super(DecoderBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, padding=padding,
                                       stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(num_features=out_ch, momentum=0.9)

    def forward(self, ten):
        ten = self.conv(ten)
        ten = self.bn(ten)
        ten = F.relu(ten, True)
        return ten


class Encoder(nn.Module):
    def __init__(self, in_ch=3, z_size=256):
        super(Encoder, self).__init__()
        self.size = in_ch
        layer_list = []
        layer_list.append(EncoderBlock(in_ch=self.size, out_ch=64, kernel_size=3, padding=1, stride=1))
        layer_list.append(EncoderBlock(in_ch=64, out_ch=128, kernel_size=3, padding=1, stride=1))
        layer_list.append(EncoderBlock(in_ch=128, out_ch=256, kernel_size=3, padding=1, stride=1))
        layer_list.append(EncoderBlock(in_ch=256, out_ch=256, kernel_size=3, padding=1, stride=1))
        layer_list.append(EncoderBlock(in_ch=256, out_ch=256, kernel_size=4, padding=1, stride=2))

        # calcuration image size by cnn per every layer : out_size = (in_size + (2 * padding) - filter_size) / stride + 1
        self.conv = nn.Sequential(*layer_list)
        self.fc = nn.Sequential(nn.Linear(in_features=128 * 128 * 256, out_features=1024, bias=False),
                                nn.BatchNorm1d(num_features=1024, momentum=0.9),
                                nn.ReLU(True))
        self.mu = nn.Linear(in_features=1024, out_features=z_size)
        self.var = nn.Linear(in_features=1024, out_features=z_size)

    def forward(self, ten):
        ten = self.conv(ten)
        ten = ten.view(len(ten), -1)
        ten = self.fc(ten)
        mu = self.mu(ten)
        logvar = self.var(ten)
        return mu, logvar

    def __call__(self, *args, **kwargs):
        return super(Encoder, self).__call__(*args, **kwargs)


class Decoder(nn.Module):
    def __init__(self, z_size, size):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(in_features=z_size, out_features=1024)
        self.fc2 = nn.Sequential(nn.Linear(in_features=1024, out_features=128 * 128 * size, bias=False),
                                nn.BatchNorm1d(num_features=16 * 16 * size, momentum=0.9),
                                nn.ReLU(True))
        self.size = size
        layers_list = []
        layers_list.append(DecoderBlock(in_ch=self.size, out_ch=256, kernel_size=3, padding=1, stride=1))
        layers_list.append(DecoderBlock(in_ch=256, out_ch=256, kernel_size=3, padding=1, stride=1))
        layers_list.append(DecoderBlock(in_ch=256, out_ch=128, kernel_size=3, padding=1, stride=1))
        layers_list.append(DecoderBlock(in_ch=128, out_ch=64, kernel_size=3, padding=1, stride=1))
        layers_list.append(DecoderBlock(in_ch=64, out_ch=1, kernel_size=3, padding=1, stride=1))
        layers_list.append(nn.Sequential(
            nn.Conv2d(in_channels=self.size, out_channels=1, kernel_size=4, stride=1, padding=2),
            nn.Tanh()
        ))

        self.conv = nn.Sequential(*layers_list)

    def forwars(self, ten):
        ten = self.fc(ten)
        ten = ten.view(len(ten), -1, 16, 16)
        ten = self.conv(ten)
        return ten

    def __call__(self, *args, **kwargs):
        return super(Decoder, self).__call__(*args, **kwargs)


class Discriminator(nn.Module):
    def __init__(self, in_ch=1, recon_level=3):
        super(Discriminator, self).__init__()
        self.size = in_ch
        self.recon_level = recon_level
        layers_list = []
        layers_list.append(nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        ))
        layers_list.append(EncoderBlock(in_ch=32, out_ch=128, kernel_size=3, padding=1, stride=1))
        layers_list.append(EncoderBlock(in_ch=128, out_ch=256, kernel_size=3, padding=1, stride=1))
        layers_list.append(EncoderBlock(in_ch=256, out_ch=256, kernel_size=3, padding=1, stride=1))
        self.conv = nn.Sequential(*layers_list)
        self.fc = nn.Sequential(
            nn.Linear(in_features=16 * 16 * self.size, out_features=512, bias=False),
            nn.BatchNorm1d(num_features=512, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=1))

    def forward(self, ten, other_ten, mode='REC'):
        if mode == "REC":
            ten = torch.cat((ten, other_ten), 0)
            for i, lay in enumerate(self.conv):
                if i == self.recon_level:
                    ten, layer_ten = lay(ten, True)
                    layer_ten = layer_ten.view(len(layer_ten), -1)
                    return layer_ten
                else:
                    ten = lay(ten)
        else:
            ten = torch.cat((ten, other_ten), 0)
            for i, lay in enumerate(self.conv):
                ten = lay(ten)

            ten = ten.view(len(ten), -1)
            ten = self.fc(ten)
            return F.sigmoid(ten)

    def __call__(self, *args, **kwargs):
        return super(Discriminator, self).__call__(*args, **kwargs)


class VaeGan(nn.Module):
    def __init__(self, z_size=256, recon_level=3, device="cpu"):
        super(VaeGan, self).__init__()
        self.device = device
        self.z_size = z_size
        self.encoder = Encoder(z_size=self.z_size)
        self.decoder = Decoder(z_size=self.z_size, size=self.encoder.size)
        self.discriminator = Discriminator(in_ch=1, recon_level=recon_level)
        self.init_parameters()

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                if hasattr(m, "weight") and m.weight is not None and m.weight.requires_grad:
                    scale = 1.0 / np.sqrt(np.prod(m.weight.shape[1:]))
                    scale /= np.sqrt(3)
                    nn.init.uniform(m.weight, -scale, scale)
                if hasattr(m, "bias") and m.bias is not None and m.bias.requires_grad:
                    nn.init.constant(m.bias, 0.0)

    def forward(self, ten, gen_size=10):
        if self.training:
            ten_original = ten
            mus, logvar = self.encoder(ten)
            std = logvar.mul(0.5).exp_()
            ten_from_normal = torch.randn(len(ten), self.z_size).to(device=self.device)
            ten = ten_from_normal * std * mus
            ten = self.decoder(ten)
            ten_layer = self.discriminator(ten, ten_original, "REC")
            ten_from_normal = torch.randn(len(ten), self.z_size).to(device=self.device)
            ten = self.decoder(ten_from_normal)
            ten_class = self.discriminator(ten_original, ten, "GAN")
            return ten, ten_class, ten_layer, mus, logvar
        else:
            if ten is None:
                ten = torch.randn(gen_size, self.z_size).to(device=self.device)
                ten = self.decoder(ten)
            else:
                mus, logvar = self.encoder(ten)
                std = logvar.mul(0.5).exp_()
                ten_from_normal = torch.randn(len(ten), self.z_size)
                ten = ten_from_normal * std * mus
                ten = self.decoder(ten)
            return ten

    def __call__(self, *args, **kwargs):
        return super(VaeGan, self).__call__(*args, **kwargs)

    @staticmethod
    def loss(ten_original, ten_predicted, layer_original, layer_predicted, layer_sampled, labels_original, labels_predicted, labels_sampled, mus, variances):

        """
        :param ten_original: original images
        :param ten_predicted:  predicted images (output of the decoder)
        :param layer_original:  intermediate layer for original (intermediate output of the discriminator)
        :param layer_predicted: intermediate layer for reconstructed (intermediate output of the discriminator)
        :param labels_original: labels for original (output of the discriminator)
        :param labels_predicted: labels for reconstructed (output of the discriminator)
        :param labels_sampled: labels for sampled from gaussian (0,1) (output of the discriminator)
        :param mus: tensor of means
        :param variances: tensor of diagonals of log_variances
        :return:
        """
        # reconstruction error, not used for the loss but useful to evaluate quality
        nle = 0.5 * (ten_original.view(len(ten_original), -1) - ten_predicted.view(len(ten_predicted), -1)) ** 2
        # kl-divergence
        kl = -0.5 * torch.sum(-variances.exp() - torch.pow(mus, 2) + variances + 1, 1)
        # mse between intermediate layers for both
        mse_1 = torch.sum(0.5 * (layer_original - layer_predicted) ** 2, 1)
        mse_2 = torch.sum(0.5 * (layer_original - layer_sampled) ** 2, 1)
        # bce for decoder and discriminator for original,sampled and reconstructed
        # the only excluded is the bce_gen_original
        bce_dis_original = -torch.log(labels_original + 1e-3)
        bce_dis_sampled = -torch.log(1 - labels_sampled + 1e-3)
        bce_dis_recon = -torch.log(1 - labels_predicted + 1e-3)
        # bce_gen_original = -torch.log(1-labels_original + 1e-3)
        bce_gen_sampled = -torch.log(labels_sampled + 1e-3)
        bce_gen_recon = -torch.log(labels_predicted + 1e-3)
        '''
        bce_gen_predicted = nn.BCEWithLogitsLoss(size_average=False)(labels_predicted,
                          Variable(torch.ones_like(labels_predicted.data).cuda(), requires_grad=False))
        bce_gen_sampled = nn.BCEWithLogitsLoss(size_average=False)(labels_sampled,
                             Variable(torch.ones_like(labels_sampled.data).cuda(), requires_grad=False))
        bce_dis_original = nn.BCEWithLogitsLoss(size_average=False)(labels_original,
                         Variable(torch.ones_like(labels_original.data).cuda(), requires_grad=False))
        bce_dis_predicted = nn.BCEWithLogitsLoss(size_average=False)(labels_predicted,
                          Variable(torch.zeros_like(labels_predicted.data).cuda(), requires_grad=False))
        bce_dis_sampled = nn.BCEWithLogitsLoss(size_average=False)(labels_sampled,
                            Variable(torch.zeros_like(labels_sampled.data).cuda(), requires_grad=False))
        '''
        return nle, kl, mse_1, mse_2, \
               bce_dis_original, bce_dis_sampled, bce_dis_recon, bce_gen_sampled, bce_gen_recon