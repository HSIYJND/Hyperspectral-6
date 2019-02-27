import numpy as np
import os
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.autograd import Variable


def load_mat(data_path, label_path, test_rate):
    data_name = os.path.basename(data_path).split('.')[0]
    data_name = data_name[0].lower() + data_name[1:]
    label_name = os.path.basename(label_path).split('.')[0]
    label_name = label_name[0].lower() + label_name[1:]
    input_image = loadmat(data_path)[data_name]
    label = loadmat(label_path)[label_name]
    label = label.reshape((label.shape[0]*label.shape[1], 1))
    d1, d2, d3 = input_image.shape
    train_data = np.zeros((d1 * d2, d3))
    index = 0
    for i in range(d1):
        for j in range(d2):
            train_data[index, :] = input_image[i, j, :]
            index += 1
    # remove background pixel
    not_0 = []
    for i in range(len(label)):
        if label[i][0] != 0:
            not_0.append(i)
    train_data = train_data[not_0]
    label = label[not_0] - 1
    train_data = StandardScaler().fit_transform(train_data)
    train_x, test_x, train_y, test_y = train_test_split(train_data, label, test_size=test_rate, random_state=42)
    return train_x, test_x, train_y, test_y


def prepare_torch_data(train_x, test_x, train_y, test_y, use_cuda=False, add_axis=False):
    if add_axis:
        train_x = train_x[:, np.newaxis, :]
        test_x = test_x[:, np.newaxis, :]
    if use_cuda:
        train_x = Variable(torch.FloatTensor(train_x)).cuda()
        test_x = Variable(torch.FloatTensor(test_x)).cuda()
        train_y = Variable(torch.LongTensor(np.squeeze(np.asarray(train_y, dtype=np.int64)))).cuda()
        test_y = Variable(torch.LongTensor(np.squeeze(np.asarray(test_y, dtype=np.int64))))
    else:
        train_x = Variable(torch.FloatTensor(train_x))
        test_x = Variable(torch.FloatTensor(test_x))
        train_y = Variable(torch.LongTensor(np.squeeze(np.asarray(train_y, dtype=np.int64))))
        test_y = Variable(torch.LongTensor(np.squeeze(np.asarray(test_y, dtype=np.int64))))
    return train_x, test_x, train_y, test_y


def prepare_batch_data(train_x, test_x, train_y, test_y, use_cuda=False, add_axis=False):
    if add_axis:
        train_x = train_x[:, np.newaxis, :]
        test_x = test_x[:, np.newaxis, :]
    if use_cuda:
        train_x = torch.FloatTensor(train_x).cuda()
        test_x = Variable(torch.FloatTensor(test_x)).cuda()
        train_y = torch.LongTensor(np.squeeze(np.asarray(train_y, dtype=np.int64))).cuda()
        test_y = Variable(torch.LongTensor(np.squeeze(np.asarray(test_y, dtype=np.int64))))
    else:
        train_x = torch.FloatTensor(train_x)
        test_x = Variable(torch.FloatTensor(test_x))
        train_y = torch.LongTensor(np.squeeze(np.asarray(train_y, dtype=np.int64)))
        test_y = Variable(torch.LongTensor(np.squeeze(np.asarray(test_y, dtype=np.int64))))
    return train_x, test_x, train_y, test_y


def get_binary_data(data, label, selected_labels):
    res_data = []
    res_label = []
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            if label[i][j] in selected_labels:
                res_data.append(data[i, j, :])
                res_label.append(selected_labels[label[i][j]])
    return np.array(res_data), np.array(res_label)


if __name__ == '__main__':
    data_path = './data2/PaviaU.mat'
    label_path = './data2/PaviaU_gt.mat'

    train_x, test_x, train_y, test_y = load_mat(data_path, label_path, 0.25)
    print(train_x.shape)
    print(test_x.shape)
    print(train_y.shape)
    print(test_y.shape)


