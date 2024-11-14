from perceptron import Percep
import numpy as np
import csv
import pandas as pd

if __name__ == "__main__":

    # Imports
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    datatrain = pd.read_csv('ex04-data/train.csv', dtype=float)
    y_train= datatrain.iloc[:,0]
    X_train= datatrain.iloc[:,1:]
    datatest= pd.read_csv('ex04-data/test.csv', dtype=float)
    y_test= datatest.iloc[:,0]
    X_test= datatest.iloc[:,1:]
    
    X_train = X_train.to_numpy(dtype=float)
    y_train = y_train.to_numpy(dtype=float)
    X_test = X_test.to_numpy(dtype=float)
    y_test = y_test.to_numpy(dtype=float)

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy


    p = Percep(learning_rate=0.01, n_iters=1000)
    p.fit(X_train, y_train)
    predictions = p.predict(X_test)

    print("Perceptron classification accuracy", accuracy(y_test, predictions))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train)

    x0_1 = np.amin(X_train[:, 0])
    x0_2 = np.amax(X_train[:, 0])

    x1_1 = (-p.weights[0] * x0_1 - p.bias) / p.weights[1]
    x1_2 = (-p.weights[0] * x0_2 - p.bias) / p.weights[1]

    ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")

    ymin = np.amin(X_train[:, 1])
    ymax = np.amax(X_train[:, 1])
    ax.set_ylim([ymin - 3, ymax + 3])

    plt.show()



