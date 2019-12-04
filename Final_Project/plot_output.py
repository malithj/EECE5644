import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as colors
import numpy as np
import pandas as pd


def plot_pca_contribution(x, y, variance, num_components):
    """
    Plots the contribution of PCA components towards variance ratio
    :return:
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    plt.plot(x, y, color='blue')
    plt.axhline(variance, linestyle='--', color='red')
    plt.axvline(num_components, linestyle='--', color='red')
    plt.xticks(np.append(np.append(np.arange(0, 11, 5), 11), np.arange(15, 50, 5)),
               np.append(np.append(np.append(np.arange(0, 10, 5), ' '), 11), np.arange(15, 50, 5)))
    plt.yticks(np.arange(0, 1.1, 0.1))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    plt.text(15, variance, "Variance: {0:}%".format(variance * 100))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance by Principal Components')
    plt.grid(axis='both', linestyle='--')
    plt.savefig('pca_contribution.png')
    plt.show()


def plot_pca_heatmap(components, columns):
    """
    Plots the contribution of PCA components with original features
    :return:
    """
    map_ = pd.DataFrame(components, columns=columns)
    plt.figure(figsize=(12, 6))
    fig, ax = plt.subplots()
    im = ax.imshow(map_)
    norm = colors.BoundaryNorm(boundaries=np.arange(-1, 1.2, 0.2).tolist(), ncolors=256)
    cbar = ax.figure.colorbar(im, ax=ax, cmap='RdYlGn_r', orientation='horizontal', norm=norm)
    cbar.set_ticks(np.arange(-1, 1.2, 0.2))
    cbar.ax.set_ylabel("Correlation")
    ax.set_xticks(np.arange(map_.shape[1]))
    ax.set_yticks(np.arange(map_.shape[0]))
    ax.set_xticklabels(columns)
    ax.set_yticklabels(np.arange(1, map_.shape[1] + 1, 1))
    ax.set_ylabel("Component")
    ax.set_xlabel("Feature")
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False, labelsize=7)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=90, ha="left",
             rotation_mode="anchor")
    plt.savefig('pca_contribution-heatmap.png')
    plt.show()
