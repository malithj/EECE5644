import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_yardage():
    """
    Plots the distribution of Yardage
    :return:
    """
    train_file_path = '../../nfl-big-data-bowl-2020/train.csv'
    df = pd.read_csv(train_file_path, header=1)
    yards = df.iloc[:, 31]
    y, x, patches = plt.hist(yards.astype(int), bins=np.arange(-25, 99, 1), density=True, color='blue', edgecolor='k')
    mean = np.mean(yards)
    print(mean)
    plt.axvline(mean, linestyle='--', color='red')
    plt.text(mean + 2, 0.12, "Mean: {0:.2f}".format(mean))
    plt.xlabel('Yardage')
    plt.ylabel('Probability')
    plt.title('Distribution of Yardage in Training Data')
    plt.savefig('histogram.png')
    plt.show()