from __future__ import print_function
import os
import numpy as np
import argparse
from datetime import datetime
import time

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

from PIL import Image
import matplotlib.pylab as plt


result_dir = "./result_cell_image/result_{}".format(datetime.now().strftime("%Y%m%d"))

parser = argparse.ArgumentParser(description='Capture cell features')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='input batch size for training (default=5)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default=100)')
parser.add_argument('----no_cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--datasets', '-d', default='./traindata1',
                    help='datasets directory(./train/Hela/h0001.jpg)')
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

class make_dataset(Dataset):
    def __init__(self, dir, list, range):
        print(" Load start ")
        print("from : {}, range : {} ".format(dir, range))

        self.dataset = []
        self.len = len(os.listdir(dir))
        for i in range(len(list)):
            for j in range(range[0], range[1]):
                full_path = dir + '/' + list[i] + '/c{0:04d}.png'.format(j)
                img = Image.open(full_path)
                label = list[i]
                img = np.asarray(img)
                img = np.asarray(img).astype("f").transpose(2, 0, 1) / 128.0 - 1.0
                self.dataset.append((img, label))
        print(" Load done")

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


f = open(args.out + '/memo.txt', 'w')
f.write(datetime.now().strftime("%Y/%m/%d  %H:%M:%S") + "\n"
        "device : {}".format(device) + "\n"
        "epoch : {}".format(args.epochs) + "\n"
        "batch size : {}".format(args.batch_size) + "\n"
        "latent dimensions : 50\n"
        "image size : {} x {}".format(256, 256) + "\n"
        "cells : {}".format(cell_list) + "\n"
        "dataset : train;{}, test;{}".format(len(dataset_loader), len(testdata_loader)))
f.close()


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1_mu = nn.Linear(16 * 128 * 128, 512)
        self.fc1_logvar = nn.Linear(16 * 128 * 128, 512)

        self.fc2 = nn.Linear(512, 16 * 128 * 128)
        self.up_sample = nn.UpsamplingNearest2d(scale_factor=2)
        self.deconv1 = nn.ConvTranspose2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.deconv4 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.deconv5 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.deconv6 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.deconv7 = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)

    def encode(self, x):
        a1 = F.relu(self.conv1(x))
        a2 = F.relu(self.conv2(a1))
        a3 = F.relu(self.conv3(a2))
        a4 = F.relu(self.conv4(a3))
        a5 = F.relu(self.conv5(a4))
        a6 = F.relu(self.conv6(a5))
        a7 = F.relu(self.conv7(a6))
        max_pool = self.max_pool(a7)
        a_reshaped = max_pool.reshape(-1, 16 * 128 * 128)
        a_mu = self.fc1_mu(a_reshaped)
        a_logvar = self.fc1_logvar(a_reshaped)
        return a_mu, a_logvar

    def decode(self, z):
        b1 = F.relu(self.fc2(z))
        b1 = b1.reshape(-1, 16, 128, 128)
        b1_upsample = self.up_sample(b1)
        b2 = F.relu(self.deconv1(b1_upsample))
        b3 = F.relu(self.deconv2(b2))
        b4 = F.relu(self.deconv3(b3))
        b5 = F.relu(self.deconv4(b4))
        b6 = F.relu(self.deconv5(b5))
        b7 = F.relu(self.deconv6(b6))
        b8 = torch.sigmoid(self.deconv7(b7))
        return b8

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        # print(z)
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0003)
cell_list = os.listdir(args.dataset)
dataset_loader = make_dataset(args.dataset, cell_list, range=(1, 101))
testdata_loader = make_dataset(args.dataset, cell_list, range=(101, 131)

f = open(args.out + '/memo.txt', 'w')
f.write(datetime.now().strftime("%Y/%m/%d  %H:%M:%S") + "\n"
        "device : {}".format(device) + "\n"
        "epoch : {}".format(args.epochs) + "\n"
        "batch size : {}".format(args.batch_size) + "\n"
        "latent dimensions : 50\n"
        "image size : {} x {}".format(256, 256) + "\n"
        "cells : {}".format(cell_list) + "\n"
        "dataset : train;{}, test;{}".format(len(dataset_loader), len(testdata_loader)))
f.close()

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 256, 256), reduction='sum')

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
        if epoch % 5 == 0:
            # 10エポックごとに最初のminibatchの入力画像と復元画像を保存
            if batch_idx == 0:
                n = 2
                comparison = torch.cat([data[:n], recon_batch.view(args.batch_size, 1, 256, 256)[:n]])
                save_image(comparison.data.cpu(),
                           '{}/reconstruction_{}.png'.format(args.out, epoch), nrow=n)
    test_loss /= len(testdata_loader.dataset)

    return test_loss


def main():
    print(device)
    loss_list = []
    test_loss_list = []
    for epoch in range(1, args.epochs + 1):
        t1 = time.time()
        loss = train(epoch)
        test_loss = test(epoch)
        t2 = time.time()

        print('epoch [ {} / {} ], loss:{:.4f}, test_loss:{:.4f}'.format(epoch, args.epochs, loss, test_loss))
        print('time / 1 epoch : {:.4f} sec, remaining time : {:.4f} min'.format(t2 - t1, (t2 - t1) * (args.epochs - epoch) / 60))

        loss_list.append(loss)
        test_loss_list.append(test_loss)

        if epoch % 5 == 0:
            torch.save(model.state_dict(), args.out + '/cell_vae_{}.pth'.format(epoch))

    np.save(args.out + '/loss_list.npy', np.array(loss_list))
    np.save(args.out + '/test_loss_list.npy', np.array(test_loss_list))
    torch.save(model.state_dict(), args.out + '/cell_vae.pth')

    # matplotlib
    loss_list = np.load('{}/loss_list.npy'.format(args.out))
    test_loss_list = np.load('{}/test_loss_list.npy'.format(args.out))
    plt.plot(loss_list, label="train loss")
    plt.plot(test_loss_list, label="test loss")
    plt.title('learning_curve')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.show()
    
    device = torch.device('cpu')
    model = VAE()
    model.load_state_dict(torch.load('{}/cell_vae.pth'.format(args.out), map_location=device))
    test_loader = make_dataset(args.dataset, cell_list, range=(500, 1000))
    images, labels = iter(test_loader).next()
    # images = image.view(len(test_loader), -1)
    
    with torch.no_grad():
        z = model.encode(Variable(images))
    mu, logvar = z
    mu, logvar = mu.data.numpy(), logvar.data.numpy()
    np.savetxt('{}/mu.csv'.format(args.out), mu, delimiter=',')
    np.savetxt('{}/labels.csv'.format(args.out), labels, delimiter=',')
    
    data1 = pd.read_csv('{}/mu.csv'.format(args.out), header=None, dtype=np.float64)
    data1.colums = ["C" + str(i) for i in range(1, len(data1.columns) + 1)]
    data2 = pd.read_csv('{}/labels.csv'.format(args.out), header=None, dtype=np.int64)]
    data2.columns = ["C0"]
    data = pd.concat([data2, data1], axis=1)
    fea = []
    for i in range(len(cell_list)):
        fea.clear()
        cell_fea = data[data['C0'] == cell_list[i].drop('C0', axis=1)
        fea.append(cell_fea)
    
    
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
    
    corr_list = []
    for i in range(0, len(cell_list)):
        for j in range(0, len(cell_list)):
            corr = calc_corr(data_list[i], data_list[j], i, j)
            corr_list.append(corr)

    print(np.reshape(corr_list, (len(cell_list), len(cell_list))))
    plt.figure()
    sns.heatmap(np.reshape(corr_list, (len(cell_list), len(cell_list))))
    plt.savefig(args.out + '/heat_map.png')
    plt.close('all')


if __name__ == '__main__':
    main()

