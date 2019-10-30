from __future__ import print_function
import os
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

import pylab
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default=128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('----no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--out', '-o', default='./result_dir_190415',
                    help='Directory to outout the reslut')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True)

class VAE(nn.Module):

    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 512)
        self.fc21 = nn.Linear(512, 2)  # mu(平均ベクトル)
        self.fc22 = nn.Linear(512, 2)  # logvar(分散共分散行列の対数)

        self.fc3 = nn.Linear(2, 512)
        self.fc4 = nn.Linear(512, 784)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h = self.relu(self.fc1(x))
        return self.fc21(h), self.fc22(h)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

loss_list = np.load('{}/loss_list.npy'.format(args.out))
plt.plot(loss_list)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()
plt.savefig(args.out + '/train_loss.png')
plt.show()

test_loss_list = np.load('{}/test_loss_list.npy'.format(args.out))
plt.plot(test_loss_list)
plt.xlabel('epoch')
plt.ylabel('test_loss')
plt.grid()
plt.savefig(args.out + '/test_loss.png')
plt.show()

"""
device = torch.device('cpu')
model = VAE()
model.load_state_dict(torch.load('{}/vae.pth'.format(args.out),
                                 map_location=device))
test_dataset = datasets.MNIST('./data', download=True, train=False, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10000, shuffle=False)
images, labels = iter(test_loader).next()
images = images.view(10000, -1)

# 784次元ベクトルを2次元ベクトルにencode
z = model.encode(Variable(images, volatile=True))
mu, logvar = z
mu, logvar = mu.data.numpy(), logvar.data.numpy()
print(mu.shape, logvar.shape)

plt.figure(figsize=(10, 10))
plt.scatter(mu[:, 0], mu[:, 1], marker='.', c=labels.numpy(), cmap=pylab.cm.jet)
plt.colorbar()
plt.xlim((-6, 6))
plt.ylim((-6, 6))
plt.grid()
plt.savefig(args.out + '/feature.png')
plt.show()
"""