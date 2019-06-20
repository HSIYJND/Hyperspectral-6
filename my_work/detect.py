"""
检测
"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def smf(m, t):
    """
    自适应余弦检测
    :param m: p * n, 待检测数据，p，光谱数目n
    :param t: , 目标光谱，p * 1(多个取平均)
    :return: 检测结果，n * 1
    """
    t = np.mean(t, axis=1)
    p, n = m.shape
    u = np.mean(m, axis=1).reshape(-1, 1)
    m = m - u
    t = t - u
    r_hat = np.cov(m)
    g = np.linalg.pinv(r_hat)
    tmp = np.dot(np.dot(t.T, g), t)[0]
    result = np.zeros((n, 1))
    for i in range(n):
        x = m[:, i]
        r = np.dot(np.dot(x.T, g), t)[0] / tmp[0]
        result[i] = r
    ss = MinMaxScaler()
    result = ss.fit_transform(result)
    result = result.reshape(1, -1)
    return result


def msd(m, b, t):
    t = np.mean(t, axis=1).reshape(-1, 1)
    p, n = m.shape
    I = np.eye(p)
    e = np.concatenate([b, t], axis=1)
    pb = I - np.dot(b, np.linalg.pinv(b))
    pz = I - np.dot(e, np.linalg.pinv(e))
    result = np.zeros((n, 1))
    for i in range(n):
        x = m[:, i]
        r = np.dot(np.dot(x.T, pb), x) / np.dot(np.dot(x.T, pz), x)
        result[i] = r
    ss = MinMaxScaler()
    result = ss.fit_transform(result)
    result = result.reshape(1, -1)
    return result


if __name__ == '__main__':
    m = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])
    t = np.array([
        [1],
        [2]
    ])
    print(msd(m, m, t))
