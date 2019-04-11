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
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('----no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--out', '-o', default='./result_dir_190411',
                    help='Directory to outout the reslut')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()


def make_directory(path):
    if not os.path.exists(path):
        os.mkdir(path)

make_directory(args.out)
make_directory('data')

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


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def loss_function(recon_x, x, mu, logvar):
    # size_average=Falseなのでバッチ内のサンプルの合計lossを求める
    # reconstruction loss 入力画像をどのくらい正確に復元できたか？
    # 数式では対数尤度の最大化だが交差エントロピーlossの最小化と等価
    recon = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # 潜在空間zに対する正則化項
    # P(z|x) が N(0, I)に近くなる（KL-distanceが小さくなる）ようにする
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon + kld


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
        """
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))
                """

    # loss_function() は平均ではなく全サンプルの合計lossを返すのでサンプル数で割る
    train_loss /= len(train_loader.dataset)

    return train_loss


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(test_loader):
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

        test_loss /= len(test_loader.dataset)

    return test_loss


if __name__ == '__main__':
    print(device)
    loss_list = []
    test_loss_list = []
    for epoch in range(1, args.epochs + 1):
        loss = train(epoch)
        test_loss = test(epoch)

        print('epoch [{}/{}], loss: {:.4f} test_loss: {:.4f}'.format(
            epoch,
            args.epochs,
            loss,
            test_loss))

        # logging
        loss_list.append(loss)
        test_loss_list.append(test_loss)

    np.save(args.out + '/loss_list.npz', np.array(loss_list))
    np.save(args.out + '/test_loss_list.npz', np.array(test_loss_list))
    torch.save(model.state_dict(), args.out + '/vae.pth')

    # matplotlib inline
    loss_list = np.load('{}/loss_list.npz'.format(args.out))
    test_loss_list = np.load('{}/test_loss_list.npz'.format(args.out))
    plt.plot(loss_list, test_loss_list)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.show()

    model.load_state_dict(torch.load('{}/vae.pth'.format(args.out),
                                     map_location=lambda storage,
                                     loc: storage))
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
    plt.show()
