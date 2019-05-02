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

elif algo == 3:
    """
    for i in range(10):
        for j in range(10):
    Centroids = fn.centroids(Training)
    TTT = fn.TTT(Centroids,Test)
    """
    
    Centroids = fn.centroids(Training)
    img = Test[5][259]
    img2 = fn.Translation(img, 4)
    #T = fn.diff_rotate(img)
    #fn.Afficher(T)
    #T = np.matrix(T).transpose()
    #print(fn.find_min(img,T,img2,fn.diff_rotate))
    transfo = [fn.diff_x,fn.diff_y,fn.diff_rotate,fn.diff_scaling,fn.diff_PHT,fn.diff_DHT,fn.diff_thickening]
    print(fn.TTT2(Centroids,Test,transfo))

else:
    Centroids = fn.centroids(Training)
    Test1 = fn.randomize_database(Test,3,np.pi/6,0.2,1)
    Test2 = fn.randomize_database(Test,3,np.pi/6,0.2,2)
    Test3 = fn.randomize_database(Test,3,np.pi/6,0.2,3)
    transfo = [fn.diff_x,fn.diff_y,fn.diff_rotate,fn.diff_scaling,fn.diff_PHT,fn.diff_DHT,fn.diff_thickening]
    print("test base normale")
    print(fn.TTT2(Centroids,Test,transfo))
    print("test base randomize 1")
    print(fn.TTT2(Centroids,Test1,transfo))
    print("test base randomize 1")
    print(fn.TTT2(Centroids,Test2,transfo))
    print("test base randomize 1")
    print(fn.TTT2(Centroids,Test3,transfo))
    
    report = fn.testNorm(Test2,Centroids,3)
    fn.Report(report,1,write = True)

"""
else :
    
    img = Test[5][26]
    fn.Afficher(img)
    img3 = fn.Thickening(img,1,thicken = True)
    fn.Afficher(img3)
    print(img - img3)

    
"""


