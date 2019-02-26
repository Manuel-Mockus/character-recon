import scipy.io as sio
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt
import sys
###########################################
#Variables Globaux:

POURCENT = 0.8

###########################################
#lecture Base de donees

if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    filename = 'mnist-original.mat'

# Ajouter: lecture test si flag

print('lecture database : ',filename)
mat = sio.loadmat(filename)
data = np.transpose(mat['data'])
label = np.array(mat['label'])
label = label.astype(int)

I = [[]for i in range(10)]
for i in range(len(data)):
    I[label[0][i]].append(255-data[i])

#Separation Base de donee en Training/ Test


def Separation(I):
    Training = [[]for i in range(10)]
    Test = [[]for i in range(10)]
    for i in range(10):
        for j in range(len(I[i])):
            if j < len(I[i])*POURCENT:
                Training[i].append(I[i][j])
            else:
                Test[i].append(I[i][j])

    return Training,Test



def Afficher(vect):
    V = vect.reshape((28,28))
    plt.imshow(V, cmap = 'gray',vmin = 0 ,vmax = 255)
    plt.axis('off')
    plt.draw() # ne bloque pas la fenetre comme plt.show

        
    
# Calcul des Moyennes de chaque chiffre sur l'ensemble des donees dediees au Training
def centroids(Training):
    L = [np.zeros(28*28) for i in range(10)]
    for i in range(10):
        for j in range(len(Training[i])):
            L[i] = np.add(L[i],Training[i][j])
        L[i] = L[i]/len(Training[i])
        L[i] = L[i].astype(int) 
    return L

# Test d'une image: retourne le chiffre le plus proche selon la norme de 'MIK....' N
def test(Image,Centroids,N):
    return np.argmin([np.linalg.norm(Centroids[i]-Image,N) for i in range(10)]) 


# Effectue un Test pour chaque image de l'ensemble 'Test' et verifie le resultat
# Retourne une liste avec les pourcentages d'identification correcte pour chaque chiffre
def pourcentage(Test,Centroids,N):
    P = [0]*10
    for i in range(10):
        for j in range(len(Test[i])):
            if test(Test[i][j],Centroids,N) == i:
                P[i] += 1
        P[i]/=len(Test[i])
    return P

# Effectue les test pour la norme inf, et les normes-p pour p in [1,20], puis graphique les resultats
def testNorm(Test,Centroids,Nb):
    pourcentages=[]
    #Report_p contient les resultats des pourcentages
    Report_p = []
    k = pourcentage(Test,Centroids,np.inf)
    M = sum(k)/10
    k.append(M)
    Report_p.append(k)
    pourcentages.append(M)
    for i in range(1,Nb):
        k = (pourcentage(Test,Centroids,i))
        print('Processing Norm :',i ,"Out of",Nb)
        #print(k)
        M = sum(k)/10
        k. append(M)
        Report_p.append(k)
        pourcentages.append(M)

    #plt.bar([i for i in range(Nb)],pourcentages)
    #plt.show()
    return Report_p


#Report dans le terminal des pourcentages par chiffre et par norme
def Report(Report_p):
    N = len(Report_p)
    R = Report_p.copy()
    for i in range(len(Report_p)):
        for j in range(len(Report_p[i])):
            R[i][j] = '%.5f'%Report_p[i][j]

    print('Percentages of correctly identified digits for each norm')
    print('Norm   |   0   |   1   |   2   |   3   |   4   |   5   |   6   |   7   |   8   |   9   |   total')
    print('inf    ', *R[0] , sep ="|")
    for i in range(1,len(R)):
        print(i,'     ', *R[i] , sep ="|")


#SVD
def svd_base(training) :
    bases = [[] for i in range(10)]
    for i in range(10) :
        A = np.matrix(np.vstack([training[i][j] for j in range(len(training[i]))])).transpose()
        bases[i] = np.linalg.svd(A)[0]
    return bases

def test_svd(image,M_k,threshold) :#à corriger
    least_squares = [np.linalg.norm(np.matmul(M_k[i],np.array([image]).transpose()),2) for i in range(10)]
    k = np.argmin(least_squares)
    min_1 = least_squares.pop(k)
    min_2 = np.min(least_squares)
    if(min_1 > min_2*threshold):
        return 10
    return k


def calcul_M_k(bases_svd,k):
    bases_k = [bases_svd[i][:,:k] for i in range(10)]
    return [np.identity(28*28)-np.matmul(bases_k[i],bases_k[i].transpose()) for i in range(10)]


                     
def split(I,train_prop) :
    training = [[] for i in range(10)]    
    test = [[]for i in range(10)]
    for i in range(10) :
        l = len(I[i])
        nb = int(train_prop*l)
        training[i] = I[i][:3]#indice à changer
        test[i] = I[i][nb:]
    return training,test

def pourcentage_SVD(Test,M_k,threshold):
    P1 = [0]*10
    P2 = [0]*10
    for i in range(10):
        print("chifre en traitement: ",i)
        for j in range(len(Test[i])):
            T = test_svd(Test[i][j],M_k,threshold)
            if T == i:
                P1[i] += 1
            elif T == 10:
                P2[i] += 1
            
        P1[i]/=(len(Test[i])-P2[i])
        P2[i]/= len(Test[i])
    return P1,P2

################################
########### MAIN ###############
################################
Training,Test = Separation(I)

'''
###############"""""
# Test Algo1
L = centroids(Training)

Afficher(I[2][3])

Report_p = testNorm(I,L,5)

Report(Report_p)
#####################"
'''

bases = svd_base(Training)
img = Test[3][302]
M_k = calcul_M_k(bases,5)           
P1,P2 = pourcentage_SVD(Test,M_k,0.95)
print("P1",P1)
print(sum(P1)/len(P1))
print("P2",P2)
print(sum(P2)/len(P2))

'''
plt.imshow(img.reshape((28,28)),cmap='gray')     
plt.show()
#print(test_svd(test_set[0][0],bases))
'''
