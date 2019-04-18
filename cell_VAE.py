from __future__ import print_function
import os
import numpy
import sys
import argparse

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from torch.nn import functional as H
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset

import pylab
import matplotlib.pylab as plt


parser = argparse.ArgumentParser(description='Capture cell features')
parser.add_argument('--batch_size', type=int, default=4, metavar='N',
                    help='input batch size for training (default=4)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default=100)')
parser.add_argument('----no_cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--datasets', '-d', default='./train',
                    help='datasets directory(./train/Hela/h0001.jpg)')
parser.add_argument('--val', '-v', default='./val',
					help='Directory to validation')
parser.add_argument('--out', '-o', default='./result_190417',
                    help='Directory to output the result')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default=1)')
parser.add_argument('log_interval' type=int, default=10, metavar='N'
                    help='How many batches to wait before logging training status')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

if not os.path.exisits(args.out):
    os.mkdir(args.out)

torch.mamual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers' : 1, 'pin_memory' : True} if args.cuda else{}

data_transform = transforms.Compose([
	transforms.RandomCrop(128),
	transforms.ToTensor()
])
test_transform = transforms.Compose([
	transforms.CenterCrop(128),
	transforms.ToTensor()
])

cell_dataset = datasets.ImageFolder(root=args.datasets,transform=data_transform)
dataset_loader = torch.utils.data.DataLoader(cell_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

cell_testdata = datasets.ImageFolder(root=args.val, transform=test_transform)
testdata_loader = torch.utils.data.DataLoader(cell_testdata, batch_size=args.batch_size, shuffle=True)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

	train_loss /= len(test_loader.dataset)




