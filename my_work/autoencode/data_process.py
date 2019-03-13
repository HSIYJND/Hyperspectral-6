import numpy as np
import random
from autoencoder.discriminant_method import l2_error


def select_background(img, gt, selected_background, split_rate=0.8):
    random.seed(42)
    data_0 = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if gt[i][j] == selected_background:
                    data_0.append(img[i][j])
    random.shuffle(data_0)
    train_split = int(len(data_0) * split_rate)
    train_data = data_0[:train_split]
    test_data_0 = data_0[train_split:]
    return np.asarray(train_data, dtype=np.float64), test_data_0


def get_testset(img, gt, test_data_0, selected_target):
    label_0 = [0] * len(test_data_0)
    data_1 = []
    label_1 = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if gt[i][j] == selected_target:
                    data_1.append(img[i][j])
                    label_1.append(1)
    return np.asanyarray(test_data_0 + data_1, dtype=np.float64), label_0 + label_1


def normalized_train(train_data):
    n_bands = train_data.shape[1]
    max_list = []
    min_list = []
    for band in range(n_bands):
        min_v = train_data[:, band].min()
        min_list.append(min_v)
        max_v = train_data[:, band].max()
        max_list.append(max_v)
        train_data[:, band] = (train_data[:, band] - min_v) / (max_v - min_v)
    return train_data, min_list, max_list


def normalized_test(test_data, min_list, max_list):
    n_bands = test_data.shape[1]
    for band in range(n_bands):
        test_data[:,band] = (test_data[:, band] - min_list[band]) / (max_list[band] - min_list[band])
    return test_data


def pre_metrics(test_label, l2_res, train_decoder, train_data, threshold_rate=0.96):
    num_1, num_test, num_0 = int(test_label.sum()), int(len(l2_res)), int(len(l2_res) - test_label.sum())
    test_0 = l2_res[: num_0]
    test_1 = l2_res[num_0:]
    train_decoder_loss = []
    for i in range(train_decoder.shape[0]):
        train_decoder_loss.append(l2_error(train_decoder[i, :], train_data[i, :]))
    train_decoder_loss.sort()
    threshold = train_decoder_loss[int(threshold_rate * len(train_decoder_loss))]
    count_0 = 0
    count_1 = 0
    for i in test_0:
        if i > threshold:
            count_0 += 1

    for i in test_1:
        if i < threshold:
            count_1 += 1

    res_metrics = {
        'accuracy': 1 - (count_0+count_1) / len(l2_res),
        'False alarm rate': count_0 / len(test_0),
        'Recognition rate': 1 - count_1 / len(test_1),
    }
    return res_metrics

