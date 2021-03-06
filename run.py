import scipy.io as sio
import numpy as np
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

elif algo == 3:    
    Centroids = fn.centroids(Training)
    img = Test[5][259]
    img2 = fn.Translation(img, 4)
    #T = fn.diff_rotate(img)
    #fn.Afficher(T)
    #T = np.matrix(T).transpose()
    #print(fn.find_min(img,T,img2,fn.diff_rotate))
    transfo = [fn.diff_x,fn.diff_y,fn.diff_rotate,fn.diff_scaling,fn.diff_PHT,fn.diff_DHT,fn.diff_thickening]
    print(fn.TTT2(Centroids,Test,transfo))



 



