import numpy as np
import matplotlib.pyplot as plt


def solve(m1, m2, std1, std2):
    a = 1 / (2 * std1 ** 2) - 1 / (2 * std2 ** 2)
    b = m2 / (std2 ** 2) - m1 / (std1 ** 2)
    c = m1 ** 2 / (2 * std1 ** 2) - m2 ** 2 / (2 * std2 ** 2) - np.log(std2 / std1)
    return np.roots([a, b, c])


def main():
    N = 100000000  # Number of samples
    bins = 1000  # Number of bins
    np.random.seed(100)  # Ensure fixed seed

    # generate two distributions
    dist1 = np.random.normal(0, 1, N)
    dist2 = np.random.normal(1, np.sqrt(2), N)

    result = solve(0, 1, 1, np.sqrt(2))

    # plot figure
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    y1_, x1_, p1 = ax.hist(dist1, bins=bins, color='red', alpha=0.7, density=True, label=r'p(X|L=1) ~ $N(0, 1)$')
    y2_, x2_, p2 = ax.hist(dist2, bins=bins, color='blue', alpha=0.6, density=True,
                           label=r'p(X|L=2) ~ $N(\mu, \sigma^{2})$')
    ax.legend()
    ax.set_title("Gaussian Class Conditioned Probability")
    ax.set_ylabel("Density")
    ax.set_xlabel("x")
    ax.axvline(x=result[0], color='k', linestyle='--')
    ax.text(result[0] - 3.0, 0.40, r'$lower={0:}$'.format(np.round(result[0], 2)))
    ax.axvline(x=result[1], color='k', linestyle='--')
    ax.text(result[1] - 3.0, 0.40, r'$upper={0:}$'.format(np.round(result[1], 2)))
    plt.savefig('results/q4.png')
    plt.show()

    s = 1
    scaling_factor = 0.5 / s
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    y1, x1 = np.histogram(dist1, bins=bins, density=True)
    y2, x2 = np.histogram(dist2, bins=bins, density=True)
    ax.bar((x1[1:] + x1[:-1]) * .5, y1 * scaling_factor,
           width=(x1[1] - x1[0]), color="red", alpha=0.7, label=r'p(L=1|x) ~ $N(0, 1)$')
    ax.bar((x2[1:] + x2[:-1]) * .5, y2 * scaling_factor,
           width=(x2[1] - x2[0]), color="blue", alpha=0.6, label=r'p(L=2|x) ~ $N(\mu, \sigma^{2})$', )

    ax.legend()
    ax.set_title("Class Posterior Probability")
    ax.set_ylabel("Density")
    ax.set_xlabel("x")
    ax.axvline(x=result[0], color='k', linestyle='--')
    ax.text(result[0] - 1.0, 0.20, r'$lower={0:}$'.format(np.round(result[0], 2)))
    ax.axvline(x=result[1], color='k', linestyle='--')
    ax.text(result[1] - 1.0, 0.20, r'$upper={0:}$'.format(np.round(result[1], 2)))
    plt.savefig('results/q4_posterior.png')
    plt.show()


if __name__ == '__main__':
    main()
