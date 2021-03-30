import pandas as pd
import numpy as np
import random as rd
dataset = pd.read_csv('input.csv',delimiter=";", header = None)

X = dataset.iloc[2:, [0, 1]].values
rows=X.shape[0] #number of training examples
columns=X.shape[1] #number of features. Here n=2
n_iter=5
einlesen = open("input.csv",encoding='utf-8-sig')
K = einlesen.readline()
K= K.strip()
K = K.replace(";", "")
K= int(K)
Centroids=np.array([]).reshape(columns,0)
for i in range(K):
    rand=rd.randint(0,rows-1)
    Centroids=np.c_[Centroids,X[rand]]
    print("Centroids:")
    print(Centroids)

for i in range(n_iter):
    # step 2.a
    EuclidianDistance = np.array([]).reshape(rows, 0)
    for k in range(K):
        tempDist = np.sum((X - Centroids[:, k]) ** 2, axis=1)
        EuclidianDistance = np.c_[EuclidianDistance, tempDist]
    C = np.argmin(EuclidianDistance, axis=1) + 1
    print(EuclidianDistance)
    # step 2.b
    Y = {}
    for k in range(K):
        Y[k + 1] = np.array([]).reshape(2, 0)
    for i in range(rows):
        Y[C[i]] = np.c_[Y[C[i]], X[i]]

    for k in range(K):
        Y[k + 1] = Y[k + 1].T

    for k in range(K):
        Centroids[:, k] = np.mean(Y[k + 1], axis=0)
    Output = Y