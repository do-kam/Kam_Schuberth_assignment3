#https://medium.com/machine-learning-algorithms-from-scratch/k-means-clustering-from-scratch-in-python-1675d38eee42import pandas as pd
import numpy as np
import random as rd
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv('input.csv',delimiter=";", header = None)

X = dataset.iloc[2:, [0, 1]].values
rows=X.shape[0] #number of training examples
columns=X.shape[1] #number of features. Here n=2
n_iter=1
einlesen = open("input.csv",encoding='utf-8-sig')
K = einlesen.readline()
K= K.strip()
K = K.replace(";", "")
K= int(K)
Centroids=np.array([]).reshape(columns,0) #Centroid will be in 2x3format: [x1 x2 x3], [y1,y2,y3]
#Centroids is  n mal K dimentional matrix, where each column will be a centroid for one cluster
for i in range(K):
    rand=rd.randint(0,rows-1)
    Centroids=np.c_[Centroids,X[rand]]
    print("Centroids:")
    print(Centroids)

for i in range(n_iter):
    # step 2.a
    manhattendistance = np.array([]).reshape(rows, 0)
    for k in range(K):
        tempDist = np.sum(abs(X - Centroids[:, k]), axis=1) #axis1 means row, axis0 means column
        manhattendistance = np.c_[manhattendistance, tempDist]
    C = np.argmin(manhattendistance, axis=1) + 1
    print(manhattendistance)
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

#plt.scatter(X[:,0],X[:,1],c='black',label='unclustered data')
#plt.xlabel('Income')
#plt.ylabel('Number of transactions')
#plt.legend()
#plt.title('Plot of data points')
#plt.show()

#color=['red','blue','green','cyan','magenta']
#labels=['cluster1','cluster2','cluster3','cluster4','cluster5']
#for k in range(K):
#    plt.scatter(Output[k+1][:,0],Output[k+1][:,1],c=color[k],label=labels[k])
#plt.scatter(Centroids[0,:],Centroids[1,:],s=300,c='yellow',label='Centroids')
#plt.xlabel('Income')
#plt.ylabel('Number of transactions')
#plt.legend()
#plt.show()

#print("Hi")