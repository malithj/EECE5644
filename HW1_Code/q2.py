import numpy as np
import matplotlib.pyplot as plt


def main():
    # define parameters of class 1
    a_1 = 0
    b_1 = 1

    # define parameters of class 2
    a_2 = 1
    b_2 = 2

    # x axis
    x = np.arange(-10, 10, 0.5)

    # generate y function
    y = -np.divide(np.abs(x - a_1), b_1) + np.divide(np.abs(x - a_2), b_2)

    # plot figure
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.grid(which='both', color='grey', linestyle='--')
    ax.set_xticks(np.arange(-10, 11, 1))
    ax.plot(x, y, linestyle='-', color='b', marker='o')
    ax.set_title("Loglikelihood Ratio")
    ax.set_xlabel("x")
    ax.set_ylabel(r'$ln[\frac{p(x|L=1)}{p(x|L=2)}$')
    plt.savefig('results/q2.png')
    plt.show()


if __name__=='__main__':
    main()