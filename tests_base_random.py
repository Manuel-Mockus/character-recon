import scipy.io as sio
import numpy as np
import random
import matplotlib.pyplot as plt
import sys
import functions as fn

filename = 'mnist-original.mat'
I = fn.read_database(filename)

Training,Test = fn.Separation(I)

#Creation de 3 sets de tests ayant subit des transformations aléatoires (respectivement 1,2 et 3)
Test1 = fn.randomize_database(Test,3,np.pi/6,0.2,1)
Test2 = fn.randomize_database(Test,3,np.pi/6,0.2,2)
Test3 = fn.randomize_database(Test,3,np.pi/6,0.2,3)

print("Calcul des centroids pour l'algorithme 1")
Centroids = fn.centroids(Training)
print("Calcul des bases pour l'algorithme 2")
bases = fn.svd_base(Training)
M_k = fn.calcul_M_k(bases,20)
transfo = [fn.diff_x,fn.diff_y,fn.diff_rotate,fn.diff_scaling,fn.diff_PHT,fn.diff_DHT,fn.diff_thickening]
#On utilise la norme 2 pour l'algo 1, 20 vecteurs de base et un seuil de 0.95 pour l'algo 2

print("\n Comparaison des algorithmes pour une base ayant subi 1 transformation:")
print("Algorithme 1:")
print(fn.pourcentage(Test1,Centroids,2))
print("Algorithme 2:")
print(fn.pourcentage_SVD(Test1,M_k,0.95)[0])
print("Algorthme 3:")
print(fn.TTT2(Centroids,Test1,transfo))

print("\n Comparaison des algorithmes pour une base ayant subi 2 transformations:")
print("Algorithme 1:")
print(fn.pourcentage(Test2,Centroids,2))
print("Algorithme 2:")
print(fn.pourcentage_SVD(Test2,M_k,0.95)[0])
print("Algorthme 3:")
print(fn.TTT2(Centroids,Test2,transfo))

print("\n Comparaison des algorithmes pour une base ayant subi 3 transformations:")
print("Algorithme 1:")
print(fn.pourcentage(Test3,Centroids,2))
print("Algorithme 2:")
print(fn.pourcentage_SVD(Test3,M_k,0.95)[0])
print("Algorthme 3:")
print(fn.TTT2(Centroids,Test3,transfo))
