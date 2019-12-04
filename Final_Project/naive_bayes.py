from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline

from preprocess import get_preprocessed_data
from plot_output import plot_pca_contribution, plot_pca_heatmap

import numpy as np
import pandas as pd


def main():
    """
    Uses preprocessed data to perform Machine Learning Naive Bayes
    :return:
    """
    df = get_preprocessed_data()

    # digitize yards
    bins = np.linspace(-99, 100, 10)
    indices = np.digitize(df['Yards'], bins)
    df = df.assign(Yards=indices)

    # extract input column list
    input_column_list = df.columns.tolist()
    input_column_list.remove('Yards')

    # cross val
    training_x, testing_x, training_y, testing_y = train_test_split(df[input_column_list], df['Yards'], test_size=0.2, random_state = 0)

    # scale data
    scaler = preprocessing.StandardScaler()
    training_x = pd.DataFrame(scaler.fit_transform(training_x), columns=training_x.columns)
    testing_x = pd.DataFrame(scaler.fit_transform(testing_x), columns=testing_x.columns)

    # perform principal component analysis
    pca = PCA()
    pca.fit(training_x)
    explained_variance = 0.5
    pca_explained_variance = pca.explained_variance_ratio_

    # describe 50% of the variance
    pca = PCA(0.50)
    pca.fit(training_x)
    transformed_training_x = pca.transform(training_x)
    transformed_testing_x = pca.transform(testing_x)
    plot_pca_contribution(np.arange(pca_explained_variance.shape[0]), np.cumsum(pca_explained_variance),
                          explained_variance, transformed_training_x.shape[1])
    plot_pca_heatmap(pca.components_, columns=input_column_list)

    # define Gaussian Navie Bayes
    gnb = GaussianNB()
    clf = make_pipeline(preprocessing.StandardScaler(), PCA(0.50), gnb)
    scores = cross_val_score(clf, df[input_column_list], df['Yards'], cv=10)
    print("Naive Bayes Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


if __name__ == '__main__':
    main()
