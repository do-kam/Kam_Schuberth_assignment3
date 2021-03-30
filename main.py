print("assignment3")

import numpy as np
import pandas as pd


pd.set_option("display.precision", 12)
#https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
daten2 = pd.read_csv('input_form.csv',delimiter=";", header = None)
print(daten2)

#import matplotlib.pyplot as plt
#plt.figure(0)
#plt.scatter(daten2, daten2)

#aufrufen der funktion init to initialize object from class
#source: https://github.com/python-engineer/MLfromscratch/tree/master/mlfromscratch
class kmeansalgorithm():

    def __init__(self, K=3, max_iters=100, plot_steps=False):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps

        # list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K)]
        # the centers (mean feature vector) for each cluster: empty at the beginning
        self.centroids = []


#funktion um distance zu berechnen
def manhattan_distance(x1,x2):
    return np.sqrt(np.sum(abs(x1-x2)))

