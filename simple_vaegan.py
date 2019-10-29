from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image
import argparse
import time
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
import csv
from torchvision.utils import save_image


print(" This program is prototype of simple VAEGAN ")
print(" Start ")
print("==================================================")

parser = argparse.ArgumentParser(description='Cell Feature of VaeGan')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--no_cuda', action='store_true', default=False)
parser.add_argument('--dataset', default='./traindata')
parser.add_argument('--out', default='./{}'.format(datetime.now().strftime("%Y%m%d")))
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()


args.cuda = not args.no_cuda and torch.cuda.is_available()

if not os.path.exists(args.out):
    os.makedirs(args.out)

torch.manual_seed(args.seed)
device = torch.device("gpu" if args.cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
print(device)

class make_dataset(Dataset):
    def __init__(self, dir, list, range):
        print(" Load start ")
        print("from : {}, range : {} ",format(dir, range))

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


class VAE(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, z_size=256, kn=3, p=1, s=1):
        super(VAE, self).__init__()
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
        decoder_list.append(nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=kn, padding=p, stride=s))
        decoder_list.append(nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=kn, padding=p, stride=s))
        decoder_list.append(nn.ConvTranspose2d(in_channels=64, out_channels=out_ch, kernel_size=kn, padding=p, stride=s))
        self.deconv = nn.Sequential(*decoder_list)

    def forward(self, ten):
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
        return ten, mu, logvar



class GAN(nn.Module):
    def __init__(self, in_ch=3, kn=3, p=1, s=1):
        super(GAN, self).__init__()
        disc_list = []
        self.fc4 = (nn.Sequential(nn.Conv2d(in_channels=in_ch, out_channels=32, kernel_size=kn, padding=p, stride=s),
                                        nn.ReLU(True)))
        disc_list.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, padding=p, stride=2))
        disc_list.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, padding=p, stride=2))
        self.disconv = nn.Sequential(*disc_list)
        self.fc5 = nn.Sequential(nn.Linear(in_features=64*64*256, out_features=512),
                                nn.ReLU(True),
                                nn.Linear(in_features=512, out_features=1))

    def forward(self, ten, ten_original):
        ten_dis = torch.cat(self.fc4(ten), self.fc4(ten_original))
        ten_dis = self.disconv(ten_dis)
        ten_dis = self.fc5(ten_dis)
        return ten_dis


def loss_record(file_name, epoch, loss, dis_loss):
    f = open(file_name, 'a')
    writer = csv.writer(f)
    writer.writerow([epoch, loss, dis_loss])
    f.close()


model = VAE()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
dis = GAN()
dis_optimizer = optim.Adam(dis.parameters(), lr=1e-3)

cell_list = os.listdir(args.dataset)
train_loader = make_dataset(args.dataset, cell_list, range=(1, 100))
test_loader = make_dataset(args.dataset, cell_list, range(100, 140))


def loss_function(ten, ten_original, mu, logvar):
    BCE = F.binary_cross_entropy(ten.view(len(ten), 3, 256, 256), ten_original.view(len(ten_original), 3, 256, 256), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def dis_loss_function(fake, real, target, data_size=(256, 256)):
    y_in = dis(real, target)
    y_out = dis(fake, target)
    L1 = torch.sum(nn.softmax(-y_in)) / args.batch_size / data_size[0] / data_size[1]
    L2 = torch.sum(nn.softmax(y_out)) / args.batch_size / data_size[0] / data_size[1]
    return L1 + L2


def train(epoch):
    model.train()
    train_loss, train_dis_loss = 0, 0
    for batcg_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        dis_optimizer.zero_grad()
        ten, mu, logvar = model(data)
        train_loss += loss_function(ten, data, mu, logvar)
        train_dis_loss += dis_loss_function(ten, data, data)
        train_loss.backward()
        train_dis_loss.backward()
        optimizer.step()
        dis_optimizer.step()
        train_loss /= len(train_loader.dataset).item()
        train_dis_loss /= len(train_loader.dataset).item()
        return train_loss, train_dis_loss


def test(epoch):
    model.eval()
    test_loss, test_dis_loss = 0, 0
    for batch_idx, (data, _) in enumerate(test_loader):
        data = data.to(device)
        ten, mu, logvar = model(data)
        test_loss += loss_function(ten, data, mu, logvar).item()
        test_dis_loss += dis_loss_function(ten, data, data).item()
        if epoch % 10 == 0 and batch_idx == 0:
            n = 1
            comparsion = torch.cat([data[:n], ten.view(args.batch_size, 1, 300, 300)[:n]])
            save_image(comparsion.data.cpu(), '{}/reconstruction_{}.png'.format(args.out, epoch), nrow=n)
    test_loss /= len(test_loader.dataset)
    test_dis_loss /= len(test_loader.dataset)
    return test_loss, test_dis_loss


def main():
    print("device : {}".format(device))
    for epoch in range(1, args.epochs + 1):
        t1 = time.time()
        tr_loss, tr_dis_loss = train(epoch)
        te_loss, te_dis_loss = test(epoch)
        loss_record('{}/train_loss.csv'.format(args.out), epoch, tr_loss, tr_dis_loss)
        loss_record('{}/test_loss.csv'.format(args.out), epoch, te_loss, te_dis_loss)
        t2 = time.time()
        print('epoch [{} / {}], train loss : {:.4f}, trian discriminator loss : {:.4f}'.format(epoch, args.epochs, tr_loss, tr_dis_loss))
        print('                 test loss  : {:.4f}, test discriminator loss  : {:.4f}'.format(te_loss, te_dis_loss))
        print('time / 1 epoch : {:.4f} sec, remaining time : {:.4f} min'.format(t2 - t1, (t2 - t1) * (args.epochs - epoch + 1) / 60))
        if epoch % 10 == 0:
            torch.save(model.state_dict(), '{}/cell_vaegan_epoch_{}.pth'.format(args.out, epoch))
    torch.save(model.state_dict(), '{}/cell_vaegan.pth'.format(args.out))
    print(' Cell VAEGAN was finished ')


if __name__ == '__main__':
    main()
