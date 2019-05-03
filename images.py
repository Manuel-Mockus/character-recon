import scipy.io as sio
import numpy as np
import random
import matplotlib.pyplot as plt
import sys
import functions as fn
#lecture Base de donees

filename = 'mnist-original.mat'
I = fn.read_database(filename)

Training,Test = fn.Separation(I)

Centroids = fn.centroids(Training)
for i in range(10):
    fn.Afficher(Centroids[i])
