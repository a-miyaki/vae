from __future__ import print_function
import os
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

from cell_VAE import VAE

import matplotlib.pylab as plt


parser =argparse.ArgmentParser(description='VAE cell image generation')
parser.add_argument('--batch_size', type=int, default=1, metavar='N')
parser.add_argument('--epochs', type=int, default=1, metavar='N')
parser.add_argument('----no_cuda', action='store_true', default=False)
parser.add_argument('--data', default='./traindata')
parser.add_argument('--out', default='./result_cell_image/result_20190625/review')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

if not os.path.exists(args.out):
  os.mkdir(args.out)

device = torch.device("cuda" if args.cuda else "cpu")

data_transform = transforms.Compose([transforms.Garyscale(), transforms.ToTensor()])
dataset = datasets.ImageFolder(root=args.data, transform=data_transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

model = VAE().to(device)
# train:cpu != test:gpu, and, train:gpu != test:cpu

def Image_generate(epoch):
    model.eval()
    for batch_size, (data, label) in enumerate(dataset):
        data = data.to(device)
        recon_batch, _, _ = model(data)
        if epoch % == 0:
            if batch_idx == 0:
            n = len(dataset)
            comparison = torch.cat([data[:n], recon_batch.view(args.batch_size, 1, 140, 140)[:n]])
            save_image(comparison.data.cpu(), args.out + label, nrow=1)

if __name__ == '__main__':
    print(device)
    Image_generate(1)

