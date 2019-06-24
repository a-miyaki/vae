from __future__ import print_function
import os
import numpy
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

import matplotlib.pylab as plt

result_dir = "./result_{}".format(datetime.now().strftime("%Y/%m/%d")

parser = argparse.ArgumentParser(description='Capture cell features')
parser.add_argument('--batch_size', type=int, default=10, metavar='N',
                    help='input batch size for training (default=128# MINST)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default=100)')
parser.add_argument('----no_cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--datasets', '-d', default='./traindata',
                    help='datasets directory(./train/Hela/h0001.jpg)')
parser.add_argument('--val', '-v', default='./testdata',
                    help='Directory to validation')
parser.add_argument('--out', '-o', default=result_dir,
                    help='Directory to output the result')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default=1)')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='How many batches to wait before logging training status')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

if not os.path.exists(args.out):
    os.mkdir(args.out)

torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else{}

data_transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])

cell_dataset = datasets.ImageFolder(root=args.datasets, transform=data_transform)
dataset_loader = torch.utils.data.DataLoader(cell_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

cell_testdata = datasets.ImageFolder(root=args.val, transform=data_transform)
testdata_loader = torch.utils.data.DataLoader(cell_testdata, batch_size=args.batch_size, shuffle=True)

cell_list = ["HEK293", "KYSE150", "MCF-7"]


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(32, 16, 3)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1_mu = nn.Linear(68 * 68 * 16, 20)
        self.fc1_logvar = nn.Linear(68 * 68 * 16, 20)
        self.fc2 = nn.Linear(20, 68 * 68 * 16)
        self.up_sample = nn.UpsamplingNearest2d(size=(140, 140), scale_factor=2)
        self.conv3 = nn.ConvTranspose2d(16, 32, 3)
        self.conv4 = nn.ConvTranspose2d(32, 1, 3)

    def encode(self, x):
        a1 = F.relu(self.conv1(x))
        a2 = F.relu(self.conv2(a1))
        mx_poold = self.max_pool(a2)
        a_reshaped = mx_poold.reshape(-1, 68 * 68 * 16)
        a_mu = self.fc1_mu(a_reshaped)
        a_logvar = self.fc1_logvar(a_reshaped)
        return a_mu, a_logvar

    def decode(self, z):
        a3 = F.relu(self.fc2(z))
        a3 = a3.reshape(-1, 16, 68, 68)
        a3_upsample = self.up_sample(a3)
        a4 = F.relu(self.conv3(a3_upsample))
        a5 = torch.sigmoid(self.conv4(a4))
        return a5

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar = self.encode(x)
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
    for batch_idx, (data, _) in enumerate(dataset_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    train_loss /= len(dataset_loader.dataset)
    return train_loss

def test(epoch):
    model.eval()
    test_loss = 0
    for batch_idx, (data, _) in enumerate(testdata_loader):
        data = data.to(device)
        recon_batch, mu, logvar = model(data)
        test_loss += loss_function(recon_batch, data, mu, logvar).item()
        if epoch % 10 == 0:
            # 10エポックごとに最初のminibatchの入力画像と復元画像を保存
            if batch_idx == 0:
                n = 8
                comparison = torch.cat([data[:n], recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.data.cpu(),
                           '{}/reconstruction_{}.png'.format(args.out, epoch), nrow=n)
    test_loss /= len(testdata_loader.dataset)

    return test_loss

f = open(args.out + '/memo.txt', 'w')
f.write(datetime.now().strftime("%Y/%m/%d  %H:%M:%S") + "\n"
        "device : {}".format(device) + "\n"
        "epoch : {}".format(args.epochs) + "\n"
        "batch size : {}".format(args.batch_size) + "\n"
        "image size : {} x {}".format(140, 140) + "\n"
        "cells : {}, {}, {}".format("HEK293", "KYSE150", "MCF-7") + "\n"
        "dataset : train;{}, test;{}".format(350, 50))
f.close()
                                  
if __name__ == '__main__':
    print(device)
    loss_list = []
    test_loss_list = []
    for epoch in range(1, args.epochs + 1):
        loss = train(epoch)
        test_loss = test(epoch)

        print('epoch [ {} / {} ], loss:{:.4f}, test_loss:{:.4f}'.format(epoch, args.epochs, loss, test_loss))

        loss_list.append(loss)
        test_loss_list.append(test_loss)

    np.save(args.out + '/loss_list.npy', np.array(loss_list))
    np.save(args.out + '/test_loss_list.npy', np.array(test_loss_list))
    torch.save(model.state_dict(), args.out + '/cell_vae.pth')

    # matplotlib
    loss_list = np.load('{}/loss_list.npy'.format(args.out))
    test_loss_list = np.load('{}/test_loss_list.npy'.format(args.out))
    plt.plot(loss_list)
    plt.plot(test_loss_list)
    plt.title('learning_curve')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.show()

    """device = torch.device('cpu')
    model = VAE()
    model.load_state_dict(torch.load('{}/cell_vae.pth'.format(args.out), map_location=device))
    
    cell_testdata = datasets.ImageFolder(root=args.val, transform=data_transform)
    testdata_loader = torch.utils.data.DataLoader(cell_testdata, batch_size=len(testdata_loader.dataset), shuffle=False)
    images, labels = iter(testdata_loader).next()
    images = images.view(len(testdata_loader.dataset), -1)

    # 784次元ベクトルを2次元ベクトルにencode
    with torch.no_grad():
        z = model.encode(Variable(images))
    mu, logvar = z
    mu, logvar = mu.data.numpy(), logvar.data.numpy()
    print(mu.shape, logvar.shape)
    plt.scatter(mu[:, 0], mu[:, 1], marker='.', c=labels.numpy(), cmap=pylab.cm.jet)
        
    plt.figure(figsize=(7, 7))
    plt.title('Feature space')
    plt.xlim((-6, 6))
    plt.ylim((-6, 6))
    plt.grid()
    plt.show()"""
