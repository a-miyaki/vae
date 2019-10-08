import torch
import torch.utils.data
import torchvision
from torchvision import datasets, transforms
import os
from PIL import Image
import numpy as np


def random_crop(img, crop_size=256):
    _, h, w = img.shape
    x_l = np.random.randint(0, w - crop_size)
    x_r = x_l + crop_size
    y_l = np.random.randint(0, h - crop_size)
    y_r = y_l + crop_size
    return img[:, y_l:y_r, x_l:x_r]


class make_dataset(datasets):
    def __init__(self, dataDir):

        print("load dataset start")
        print("from : {}".format(dataDir))

        self.len = len(os.listdir(dataDir))
        self.data_names = [os.path.join(dataDir, name) for name in sorted(os.listdir(dataDir))]

        self.len = len(self.data_names)

    def __len__(self):
        return self.len

    def __iter__(self):
        return self

    def __getitem__(self, item):
        data = Image.open(self.data_names[item]).convert('L')
        label, _ = os.path.split(self.data_names[item])
        data = random_crop(data)
        data = data.transpose(2, 0, 1)
        data = data.astype("float32") / 128.0 - 1.0
        return (data.copy(), data.copy(), label)