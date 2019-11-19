import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.utils import resample
from sklearn import tree as tree_

import pydotplus
import collections


def main():
    # plot the data
    data = pd.read_csv('Q1.csv', header=None)
    data.columns = ['x', 'y', 'g']
    x1 = data.loc[data.g == 1].x
    y1 = data.loc[data.g == 1].y
    x2 = data.loc[data.g == -1].x
    y2 = data.loc[data.g == -1].y
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.scatter(x1, y1, c='k', marker='x', facecolor=None, label='Class 1')
    ax.scatter(x2, y2, c='red', marker='o', label='Class -1')
    ax.grid(axis='both')
    ax.legend(loc=2)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Scatter Plot of Data')
    plt.savefig('original_data.png')

    # split train and test data
    testIdx = int(data.shape[0] * 0.10)
    testData = data.iloc[0:testIdx, :]
    trainData = data.iloc[testIdx:, :]

    # train the model
    maxSplit = 12
    tree = DecisionTreeClassifier(criterion='gini', max_leaf_nodes=maxSplit, min_impurity_decrease=0.01, random_state=101).fit(trainData.iloc[:, :-1], trainData.g)
    prediction = tree.predict(testData.iloc[:, :-1])
    print("ID3 Decision Tree – The prediction accuracy is: ", tree.score(testData.iloc[:, :-1], testData.iloc[:, 2]) * 100, "%")
    cmat = confusion_matrix(testData.g, prediction, labels=[1, -1])
    accuracyScore = accuracy_score(testData.g, prediction)
    print(cmat)
    print(accuracyScore)
    data_feature_names = ['x1', 'x2']
    tree_.export_graphviz(tree, "out.dot", feature_names=data_feature_names, filled=True, rounded=True)
    graph = pydotplus.graphviz.graph_from_dot_file("out.dot")
    colors = ('turquoise', 'orange')
    edges = collections.defaultdict(list)
    for edge in graph.get_edge_list():
        edges[edge.get_source()].append(int(edge.get_destination()))

    for edge in edges:
        edges[edge].sort()
        for i in range(2):
            dest = graph.get_node(str(edges[edge][i]))[0]
            dest.set_fillcolor(colors[i])
    graph.write_png('tree.png')

    # plot decision boundaries
    x_min, x_max = data.x.min() - 1, data.x.max() + 1
    y_min, y_max = data.y.min() - 1, data.y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)

    ax.contourf(xx, yy, z, alpha=0.4)
    scatter = ax.scatter(data.x, data.y, c=data.g, s=20, edgecolor='k')
    ax.grid(axis='both')
    legend = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
    ax.add_artist(legend)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Decision Boundary of Data (Decision Tree)')
    plt.savefig('decision_boundary_dt.png')

    # bagging decision tree
    numEstimators = 7
    trainData = resample(trainData.to_numpy(), n_samples=int(data.shape[0] * 0.90), random_state=101)
    clf = RandomForestClassifier(n_estimators=numEstimators, criterion='gini', max_leaf_nodes=maxSplit, min_impurity_decrease=0.01, random_state=101)
    clf.fit(trainData[:, :-1], trainData[:, -1])
    prediction = clf.predict(testData.iloc[:, :-1])
    print("RandomForest – The prediction accuracy is: ", clf.score(testData.iloc[:, :-1], testData.iloc[:, 2]) * 100, "%")
    cmat = confusion_matrix(testData.g, prediction, labels=[-1, 1])
    accuracyScore = accuracy_score(testData.g, prediction)
    print(cmat)
    print(accuracyScore)

    # plot the decision boundary
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)

    ax.contourf(xx, yy, z, alpha=0.4)
    scatter = ax.scatter(data.x, data.y, c=data.g, s=20, edgecolor='k')
    ax.grid(axis='both')
    legend = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
    ax.add_artist(legend)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Decision Boundary of Data (Random Forrest)')
    plt.savefig('decision_boundary_rf.png')
    plt.show()

    # adaboost
    clf = AdaBoostClassifier(DecisionTreeClassifier(criterion='gini', max_leaf_nodes=maxSplit, min_impurity_decrease=0.01),
                             n_estimators=7, random_state=101, learning_rate=1.0)
    clf.fit(trainData[:, :-1], trainData[:, -1])
    print("Adaboost Weights")
    print(clf.estimator_weights_)
    prediction = clf.predict(testData.iloc[:, :-1])
    print("Adaboost – The prediction accuracy is: ", clf.score(testData.iloc[:, :-1], testData.iloc[:, 2]) * 100,
          "%")
    cmat = confusion_matrix(testData.g, prediction, labels=[-1, 1])
    accuracyScore = accuracy_score(testData.g, prediction)
    print(cmat)
    print(accuracyScore)

    # plot the decision boundary
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)

    ax.contourf(xx, yy, z, alpha=0.4)
    scatter = ax.scatter(data.x, data.y, c=data.g, s=20, edgecolor='k')
    ax.grid(axis='both')
    legend = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
    ax.add_artist(legend)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Decision Boundary of Data (Adaboost)')
    plt.savefig('decision_boundary_adb.png')
    plt.show()


if __name__ == '__main__':
    main()