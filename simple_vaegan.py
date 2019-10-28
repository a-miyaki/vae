from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import argpaser
import time
from datetime import datetime
from torch.utils.data import Dataset, DataLoder


parser = argsparse.ArgumentParser(description='Cell Feature of VaeGan')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--epochs' type=int, default=100)
parser.add_argument('--no_cuda', action='store_true', default=False)
parser.add_argument('--dataset', default='./traindata')
parser.add_argument('--out', default='./{}'.format(datetime.now().strftime("%Y%m%d")
parser.add_argument('--seed', type=int, default=1)
args.parser.parse_args()


class make_dataset(Dataset):
	def __init__(self, dir, list, range):
		print(" Load start ")
		self.dataset = []
        self.len = len(os.listdir(dataDir))
        for i in range(len(list)):
            for j in range(data_range[0], data_range[1]):
                full_path = dataDir + '/' + list[i] + '/c{0:04d}.png'.format(j)
                img = Image.open(full_path)
                label = list[i]
                img = np.asarray(img)
                img = np.asarray(img).astype("f").transpose(2, 0, 1) / 128.0 - 1.0
                self.dataset.append((img, label))

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        return self

    def __getitem__(self, i, crop_size=256):
        _, h, w = self.dataset[i][0].shape
        x_l = np.random.randint(0, w - crop_size)
        x_r = x_l + crop_size
        y_l = np.random.randint(0, h - crop_size)
        y_r = y_l + crop_size
        return self.dataset[i][0][:, y_l:y_r, x_l:x_r], self.dataset[i][1]


class VAE(nn.Module):
	def __init__(self, in_ch=3, out_ch=3, z_size=256, kn=3, p=1, s=1):
		super(VAEGAN, self).__init__()
		encoder_list = []
		encoder_list.append(nn.Conv2d(in_channels=in_ch, out_channels=64, kernel_size=kn, padding=p, stride=s))
		encoder_list.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=kn, padding=p, stride=s))
		encoder_list.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=kn, padding=p, stride=s))
		
		self.conv = nn.Sequential(*encoder_list)
		self.fc1 = nn.Sequential(nn.Linear(in_features=256*256*256, out_features=1024, bias=False),
								nn.ReLU(True))
		self.mu = nn.Linear(in_features=1024, out_features=z_size)
		self.logvar = nn.Linear(in_features=1024, out_features=z_size)
		
		self.fc2 = nn.Linear(in_features=z_size, out_features=1024)
		self.fc3 = nn.Linear(in_features=1024, out_features=256*256*256)
		decoder_list = []
		decoder_list.append(nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=kn, padding=p, stride=s)
		decoder_list.append(nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=kn, padding=p, stride=s)
		decoder_list.append(nn.ConvTranspose2d(in_channels=64, out_channels=in_ch, kernel_size=kn, padding=p, stride=s)
		self.deconv = nn.Sequential(*decoder_list)
		
		disc_list = []
		self.fc4 = (nn.Sequential(nn.Conv2d(in_channels=in_ch, out_channles=32, kernel_size=kn, padding=p, stride=s),
										nn.ReLU(Ture))
		disc_list.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, padding=p, stride=2)
		disc_list.append(nn.Conv2d(in_channles=128, out_channels=256, kernel_size=4, padding=p, stride=2)
		self.disconv = nn.Sequential(*disc_list)
		self.fc5 = nn.Sequential(nn.Linear(in_features=64*64*256, out_features=512),
								nn.ReLU(True),
								nn.Linear(in_features=512, out_features=1))
		
	def forward(self, ten, disc_ten):
		ten_original = ten
		ten = self.conv(ten)
		ten = self.fc1(ten)
		mu = self.mu(ten)
		logvar = self.logvar(ten)
		if self.training:
			std = logvar.mul(0.5).exp_()
			eps = torch.randn_like(std)
			mu = eps.mul(std).add_(mu)
		ten = self.fc2(mu)
		ten = self.fc3(ten)
		ten = self.deconv(ten)
		ten_dis = torch.cat(self.fc4(ten), self.fc4(ten_original), 0)
		ten_dis = self.disconv(ten_dis)
		ten_dis = self.fc5(ten_dis)
		return ten, mu, logvar, ten_dis


class GAN(nn.Module):
	


model = VAEGAN()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

cell_list = os.listdir(args.dataset)
train_loader = make_dataset(args.dataset, cell_list, range=(1, 100)
test_loader = make_dataset(args.dataset, cell_list range(100, 140)


def loss_function(ten, ten_original, mu, logvar):
	BCE = F.binary_cross_entropy(ten.view(len(ten), 3, 256, 256), ten_original.view(len(ten_original), 3, 256, 256), reduction='sum')
	KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
	return BCE + KLD


def dis_loss_function(fake, real, target,


def train(epoch):
	model.train()
	train_loss = 0
	for batcg_idx, (data, _) in enumerate(train_loader):
		data = data.to(device)
		optimizer.zero_grad()
		ten, mu, logvar, ten_dis = model(data)
		loss = loss_function(ten, data, mu, logvar)
		
