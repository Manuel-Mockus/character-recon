import scipy.io as sio
import numpy as np
import random
import matplotlib.pyplot as plt
import sys
import functions as fn


if len(sys.argv) == 2:
    filename = 'mnist-original.mat'
    algo = int(sys.argv[1])
else:
    print("argument error")
    sys.exit()

################################
########### MAIN ###############
################################

#lecture Base de donees
I = fn.read_database(filename)

if algo == 1:

    """
    Test sur toute la base avec 50 sets de training tirés aléatoirement
    avec la norme 2 ou  3 (donne expérimentalement les meilleurs résultats)
    """
    norme = 2
    P_total = [0]*11
    for k in range(50) :
        print("Test",k+1,":")
        Training,Test = fn.Separation(I)
        Centroids = fn.centroids(Training)
        print("processing")
        P = fn.pourcentage(Test,Centroids,norme)[0] 
        for i in range(11) :
            P_total[i]+=P[i]
    P_total = [x/50 for x in P_total]
    print("Success rate (each digits and mean):")
    print(P_total)

elif algo == 2 :

    """
    Test sur toute la base avec 30 sets de training tirés aléatoirement
    nombre de vecteurs de base : 20
    """
    seuil = 0.95
    P_total = [0]*11
    R_total = [0]*11
    for k in range(30) :
        print("Test",k+1,":")
        Training,Test = fn.Separation(I)
        bases = fn.svd_base(Training)
        print("calcul des M_k")
        M_k = fn.calcul_M_k(bases,20)
        print("processing")
        P,R,S = fn.pourcentage_SVD(Test,M_k,seuil)
        for i in range(11) :
            P_total[i]+=P[i]
            R_total[i]+=R[i]
    P_total = [x/30 for x in P_total]
    R_total = [x/30 for x in R_total]
    print("True positives rate (each digit and mean):")
    print(P_total)
    print("Rejected rate (each digit and mean):")
    print(R_total)

elif algo == 3 :

    """
    Test sur toute la base avec 10 sets de training tirés aléatoirement
    """
    P_total = [0]*11
    transfo = [fn.diff_x,fn.diff_y,fn.diff_rotate,fn.diff_scaling,fn.diff_PHT,fn.diff_DHT,fn.diff_thickening]
    for k in range(10) :
        print("Test",k+1,":")
        Training,Test = fn.Separation(I)
        Centroids = fn.centroids(Training)
        print("processing")
        P = fn.TTT2(Centroids,Test,transfo)
        for i in range(11) :
            P_total[i]+=P[i]
    P_total = [x/10 for x in P_total]
    print("Success rate (each digits and mean):")
    print(P_total)
    





