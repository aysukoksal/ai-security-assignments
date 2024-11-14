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

   
    
    def plot_training(p, X_train):
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
    
    def plot_test(p, X_test):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.scatter(X_test[:, 0], X_test[:, 1], marker="o", c=y_test)

        x0_1 = np.amin(X_test[:, 0])
        x0_2 = np.amax(X_test[:, 0])

        x1_1 = (-p.weights[0] * x0_1 - p.bias) / p.weights[1]
        x1_2 = (-p.weights[0] * x0_2 - p.bias) / p.weights[1]

        ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")

        ymin = np.amin(X_test[:, 1])
        ymax = np.amax(X_test[:, 1])
        ax.set_ylim([ymin - 3, ymax + 3])

        plt.show()
    p1= Percep().create_percep(X_train, y_train, X_test,y_test)
    p2= Percep().create_percep(X_train, y_train, X_test,y_test)
    p3= Percep().create_percep(X_train, y_train, X_test,y_test)
    p4= Percep().create_percep(X_train, y_train, X_test,y_test)
    p5= Percep().create_percep(X_train, y_train, X_test,y_test)
    p6= Percep().create_percep(X_train, y_train, X_test,y_test)
    p7= Percep().create_percep(X_train, y_train, X_test,y_test)
    p8= Percep().create_percep(X_train, y_train, X_test,y_test)
    p9= Percep().create_percep(X_train, y_train, X_test,y_test)
    p10= Percep().create_percep(X_train, y_train, X_test,y_test)
    
    perceps=[p1,p2,p3,p4,p5,
                p6,p7,p8,p9,p10]
    sorted_objects = sorted(perceps, key=lambda obj: obj.ber_score)
    lowest_ber= sorted_objects[0]
    
    print(f"Lowest ber perceptron weights: {lowest_ber.weights}")

    sorted_objects = sorted(perceps, key=lambda obj: obj.ber_score, reverse=True)
    best_ber= sorted_objects[0]
    plot_test(best_ber, X_test)
    plot_training(best_ber, X_train)
    
    
    



 



