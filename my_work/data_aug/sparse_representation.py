import numpy as np


class SparseRepresentation:
    def __init__(self):
        pass

    def __get_weight_array(self, m, n, seed=None):
        """
        生成波段稀疏表示的权值矩阵
        :param m: 要生成的数据个数
        :param n: 现有波段个数
        :param seed: 随机种子
        :return: 权值矩阵
        """
        np.random.seed(seed)
        a = np.random.random([m, n])
        a_sum = np.sum(a, axis=1)
        a_sum = np.reshape(a_sum, newshape=(m, 1))
        a_res = a / a_sum
        return a_res

    def get_more_data(self, data, m, n, seed=None):
        """
        用少量数据生成更多数据
        :param data: 少量数据矩阵（n, k），n为数据个数，k为数据特征维度
        :param m: 要生成的数据个数数据个数
        :param n: 少量数据个数
        :param seed: 随机种子
        :return: 扩充后的数据矩阵
        """
        weight_array = self.__get_weight_array(m, n, seed)
        return np.dot(weight_array, data)


if __name__ == '__main__':
    m, n, seed = 5, 3, 42
    sr = SparseRepresentation()
    data = np.array([
        [1.1, 2],
        [1, 2.1],
        [0.9, 2.1],
    ])
    print(sr.get_more_data(data, m, n, seed))
