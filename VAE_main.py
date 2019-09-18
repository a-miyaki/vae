from __future__ import print_function
import os
import numpy as np
import argparse
import seaborn as sns

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default=128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('----no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--out', '-o', default='./result_MNIST/0830_3D',
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
        self.fc21 = nn.Linear(512, 20)  # mu(平均ベクトル)
        self.fc22 = nn.Linear(512, 20)  # logvar(分散共分散行列の対数)

        self.fc3 = nn.Linear(20, 512)
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

    np.save(args.out + '/loss_list.npy', np.array(loss_list))
    np.save(args.out + '/test_loss_list.npy', np.array(test_loss_list))
    torch.save(model.state_dict(), args.out + '/vae.pth')

    # matplotlib inline
    loss_list = np.load('{}/loss_list.npy'.format(args.out))
    test_loss_list = np.load('{}/test_loss_list.npy'.format(args.out))
    plt.plot(loss_list)
    plt.plot(test_loss_list)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.show()

    device = torch.device('cpu')
    model = VAE()
    model.load_state_dict(torch.load('{}/vae.pth'.format(args.out),
                                     map_location=device))
    test_dataset = datasets.MNIST('./data', download=True, train=False, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10000, shuffle=False)
    images, labels = iter(test_loader).next()
    images = images.view(10000, -1)

    # 784次元ベクトルを2次元ベクトルにencode
    with torch.no_grad():
        z = model.encode(Variable(images))
    mu, logvar = z
    mu, logvar = mu.data.numpy(), logvar.data.numpy()
    np.savetxt(args.out + '/mu.csv', mu, delimiter=',')
    np.savetxt(args.out + '/label.csv', labels, delimiter=',')

    data1 = pd.read_csv(args.out + '/mu.csv', header=None, dtype=np.float64)
    data1.columns = ["C" + str(i) for i in range(1, len(data1.columns) + 1)]
    data2 = pd.read_csv(args.out + '/label.csv', header=None, dtype=np.int64)
    data2.columns = ["C0"]
    data = pd.concat([data2, data1], axis=1)

    fea_0 = data[data['C0'] == 0].drop('C0', axis=1)
    fea_1 = data[data['C0'] == 1].drop('C0', axis=1)
    fea_2 = data[data['C0'] == 2].drop('C0', axis=1)
    fea_3 = data[data['C0'] == 3].drop('C0', axis=1)
    fea_4 = data[data['C0'] == 4].drop('C0', axis=1)
    fea_5 = data[data['C0'] == 5].drop('C0', axis=1)
    fea_6 = data[data['C0'] == 6].drop('C0', axis=1)
    fea_7 = data[data['C0'] == 7].drop('C0', axis=1)
    fea_8 = data[data['C0'] == 8].drop('C0', axis=1)
    fea_9 = data[data['C0'] == 9].drop('C0', axis=1)
    # print(fea_1)


    def calc_corr(data1, data2, i, j):

        list = []
        
        for i in range(min(len(data1), len(data2))):
            np.random.seed(i)
            n = np.random.randint(0, min(len(data1), len(data2)))
            s0 = pd.Series(data1.iloc[n, :])
            np.random.seed(j)
            n = np.random.randint(0, min(len(data1), len(data2)))
            s1 = pd.Series(data2.iloc[n, :])

            res = s0.corr(s1)
            list.append(res)

        # print("{}vs{} : {}".format(i, j, np.average(list)))
        return np.average(list)

    data_list = [fea_0, fea_1, fea_2, fea_3, fea_4, fea_5, fea_6, fea_7, fea_8, fea_9]
    corr_list = []
    for i in range(0, 10):
        for j in range(0, 10):
            corr = calc_corr(data_list[i], data_list[j], i, j)
            corr_list.append(corr)

    print(np.reshape(corr_list, (10, 10)))
    plt.figure()
    sns.heatmap(np.reshape(corr_list, (10, 10)))
    plt.savefig(args.out + '/heat_map.png')
    plt.close('all')
