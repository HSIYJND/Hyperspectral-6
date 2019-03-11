import numpy as np
import os
from scipy.io import loadmat
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from collections import Counter
from torch.autograd import Variable
from tqdm import tqdm
import matplotlib.pyplot as plt


class AutoEncoder(nn.Module):
    def __init__(self, n_bands):
        super(AutoEncoder, self).__init__()
        self.n_bands = n_bands
        self.encoder = nn.Sequential(
            nn.Linear(self.n_bands, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
        )

        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, self.n_bands),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def cos_sim(x, y):
    return np.dot(x, y) / np.sqrt(np.square(x).sum()) * np.sqrt(np.square(y).sum())


def io_cos(test_decoder, test_data):
    cos_res = []
    for i in range(test_decoder.shape[0]):
        cos_res.append(cos_sim(test_decoder[i, :], test_data[i, :]))
    return cos_res


def train_autoencoder(model, train_data, lr, epochs):
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)
    loss_func = nn.MSELoss()

    train_x = Variable(torch.FloatTensor(train_data))
    losses = []
    for epoch in tqdm(range(epochs)):
        encoded, decoded = autoencoder(train_x)

        loss = loss_func(decoded, train_x)
        losses.append(loss.data.numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model, decoded, losses


def select_background_target(img, gt, select_dic):
    data_0 = []
    data_1 = []
    y_0 = []
    y_1 = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if gt[i][j] in select_dic:
                if select_dic[gt[i][j]] == 0:
                    data_0.append(img[i][j])
                    y_0.append(0)
                else:
                    data_1.append(img[i][j])
                    y_1.append(1)
    return np.asarray(data_0 + data_1, dtype=np.float64), np.asarray(y_0 + y_1, dtype=np.float64)


def normalized_data(data, n_bands):
    for band in range(n_bands):
        min_v = data[:, band].min()
        max_v = data[:, band].max()
        data[:, band] = (data[:, band] - min_v) / (max_v - min_v)
    return data


def pre_metrics(label, cos_res, train_decoder, threshold_rate=0.9):
    num_1, num_test, num_0 = int(label[10000:].sum()), int(len(cos_res)), int(len(cos_res) - label[10000:].sum())
    test_0 = cos_res[: num_0]
    test_1 = cos_res[num_0:]
    train_decoder_cos_res = []
    for i in range(train_decoder.shape[0]):
        train_decoder_cos_res.append(cos_sim(train_decoder[i, :], train_data[i, :]))
    train_decoder_cos_res.sort()
    threshold = train_decoder_cos_res[int(threshold_rate * len(train_decoder_cos_res))]
    count_0 = 0
    count_1 = 1
    for i in test_0:
        if i > threshold:
            count_0 += 1

    for i in test_1:
        if i < threshold:
            count_1 += 1

    res_metrics = {
        'accuracy': (count_0+count_1) / len(cos_res),
        'False alarm rate:': count_0 / len(test_0),
        'Recognition rate': 1 - count_1 / len(test_1),
    }
    return res_metrics


img = loadmat('./data/Salinas_corrected.mat')['salinas_corrected']
gt = loadmat('./data/Salinas_gt.mat')['salinas_gt']
n_bands = img.shape[-1]
data, label = select_background_target(img, gt, {8:0, 11:1})
data = normalized_data(data, n_bands)
train_data = data[:10000]
test_data = data[10000:]
autoencoder = AutoEncoder(n_bands)
print(autoencoder)

autoencoder, decoded, losses = train_autoencoder(autoencoder, train_data, 0.001, 1000)
train_decoder = decoded.data.numpy()
test_x = Variable(torch.FloatTensor(test_data))
test_encoder, test_decoder = autoencoder(test_x)
test_encoder, test_decoder = test_encoder.data.numpy(), test_decoder.data.numpy()

cos_res = io_cos(test_decoder, test_data)
res_metrics = pre_metrics(label, cos_res, train_decoder)
print(res_metrics)


