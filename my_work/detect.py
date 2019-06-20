"""
检测
"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def cov(m):
    p, n = m.shape
    u = np.mean(m, axis=1).reshape(-1, 1)
    m -= u
    return np.dot(m, m.T) / n


def ace(m, s):
    s = np.mean(s, axis=1).reshape(-1, 1)
    p, n = m.shape
    u = np.mean(m, axis=1).reshape(-1, 1)
    m = m - u
    r_hat = cov(m)
    g = np.linalg.inv(r_hat)
    result = np.zeros((n, 1))
    tmp = np.linalg.inv(np.dot(np.dot(s.T, g), s))
    gs = np.dot(g, s)
    sg = np.dot(s.T, g)
    for i in range(n):
        x = m[:, i]
        r1 = np.dot(x.T, gs)
        r2 = np.dot(sg, x)
        result[i] = np.dot(np.dot(r1, tmp), r2) / np.dot(np.dot(x.T, g), x)
    ss = MinMaxScaler()
    result = ss.fit_transform(result)
    result = result.reshape(1, -1)
    return result


def smf(m, t):
    """
    测试通过
    """
    t = np.mean(t, axis=1).reshape(-1, 1)
    p, n = m.shape
    u = np.mean(m, axis=1).reshape(-1, 1)
    m = m - u
    t = t - u
    r_hat = cov(m)
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


def msd_extract(m):
    cm = cov(m)
    v, d = np.linalg.eig(cm)
    d = np.fliplr(d)
    return d[:, :3]


def msd(m, b, t):
    b = msd_extract(b)
    t = msd_extract(t)
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
        [1., 2, 3],
        [4, 2, 1]
    ])
    t = np.array([
        [1],
        [2]
    ])
    print(cov(m))
