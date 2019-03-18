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
    """
    for i in range(10):
        for j in range(10):
    Centroids = fn.centroids(Training)
    TTT = fn.TTT(Centroids,Test)
    """
    #print(fn.TTT(Centroids,Test))
    Centroids = fn.centroids(Training)
    print(Centroids[1].shape)
    img = Test[1][203]
    img2 = fn.Translation(img, 2)
    T = np.matrix(np.diff(img2))
    print(T.shape)
    T = np.stack([T,np.array([0])])
    #T = np.matrix(T).transpose()
    print(fn.find_min_translate_x(img,T,img2))
    
    
#fn.Report(report,algo,write = True)




