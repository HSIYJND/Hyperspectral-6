import numpy as np


def cos_sim(x, y):
    return np.dot(x, y) / np.sqrt(np.square(x).sum()) / np.sqrt(np.square(y).sum())


def l2_error(x, y):
    return np.sqrt(np.square(x - y).sum())


def io_cos(test_decoder, test_data):
    cos_res = []
    for i in range(test_decoder.shape[0]):
        cos_res.append(cos_sim(test_decoder[i, :], test_data[i, :]))
    return cos_res


def io_l2_error(test_decoder, test_data):
    l2_res = []
    for i in range(test_decoder.shape[0]):
        l2_res.append(l2_error(test_decoder[i, :], test_data[i, :]))
    return l2_res

