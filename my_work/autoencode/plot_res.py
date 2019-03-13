import pandas as pd
import os
import matplotlib
from tqdm import tqdm
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root = './result/salinas/'
    paths = os.listdir(root)
    save_root = './result/salinas_plot/'

    for path in tqdm(paths):
        df = pd.read_csv(root + path)
        false_alarm_rate = df['False alarm rate'][0]
        plt.bar(range(len(df)), df['Recognition rate'])
        plt.title(path.split('.')[0] + ',false_alarm_rate:{}%'.format(round(false_alarm_rate*100, 2)))
        plt.xticks(range(len(df)), df['class'])
        plt.ylabel('Recognition rate')
        plt.xlabel('other classes')
        save_path = save_root + path.split('.')[0] + '.png'
        plt.savefig(save_path)
        plt.clf()


if __name__ == '__main__':
    main()

