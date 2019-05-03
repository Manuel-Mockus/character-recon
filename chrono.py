import scipy.io as sio
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt
import sys
import functions as fn
from time import process_time

mytime = 0
def tic():
    global mytime
    mytime = process_time()
    
def tac():
    global mytime
    print (process_time()-mytime)
    mytime = process_time()
    
filename = 'mnist-original.mat'

I = fn.read_database(filename)



Training,Test = fn.Separation(I)
Test100 = [Test[i][:100] for i in range(10)]

print("Temps calcul centroids : ")
tic()
Centroids = fn.centroids(Training)
tac()

print("Temps execution algo 1 sur 1000 images (norme 2) :")
tic()
fn.pourcentage(Test100,Centroids,2)
tac()

print("Algo 2 avec 20 vecteurs de bases et un seuil de 0.95 :")
print("Temps calcul base SVD :")
tic()
bases = fn.svd_base(Training)
M_k = fn.calcul_M_k(bases,20)
tac()

print("Temps execution algo 2 sur 1000 images :")

tic()
fn.pourcentage_SVD(Test100,M_k,0.95)
tac()


print("Temps execution algo 3 sur 1000 images :")
transfo = [fn.diff_x,fn.diff_y,fn.diff_rotate,fn.diff_scaling,fn.diff_PHT,fn.diff_DHT,fn.diff_thickening]
tic()
fn.TTT2(Centroids,Test100,transfo)
tac()
