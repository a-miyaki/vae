from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import numpy as np


class make_dataset(Dataset):
    def __init__(self, dataDir, list, data_range):
        print("load dataset start")
        print("from : {}".format(dataDir))

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


if __name__ == '__main__':
    list = ["HEK-293", "KMST-6"]
    dir = make_dataset('../testdata1', list, range(1141, 1900))
    gen = DataLoader(dir, batch_size=128, shuffle=False, num_workers=1)

