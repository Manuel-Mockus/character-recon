import scipy.io as sio
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt
import sys
import functions as fn
#lecture Base de donees

if len(sys.argv) > 2:
    filename = sys.argv[1]
    algo = int(sys.argv[2])
else:
    filename = 'mnist-original.mat'
    algo = 1

################################
########### MAIN ###############
################################

I = fn.read_database(filename)
Training,Test = fn.Separation(I)

if algo == 1:
    #Algorithme Centroides
    Centroids = fn.centroids(Training)
    report = fn.testNorm(Test,Centroids,10)
 

elif algo == 2 :
    #Algorithme SVD
    bases = fn.svd_base(Training)
    report = fn.test_bases_SVD(Test,bases,1,4)
    
    #Graphe 
    #report = fn.SVD_show_3D(Test,bases,4,8,13)
    #report = fn.SVD_show_2D(Test,bases,10,0.95,0.999)

else:
    Centroids = fn.centroids(Training)
    print(fn.TTT(Centroids,Test))
    
    
#fn.Report(report,algo,write = True)




