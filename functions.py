import scipy.io as sio
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt
import sys
###########################################
#Variables globales:


POURCENT = 0.8 #proportion choisie du set Training par rapport à la base de donnée totale

###########################################

#Separation Base de donee en Training/ Test

def read_database(filename):
    print('lecture database : ',filename)
    mat = sio.loadmat(filename)
    data = np.transpose(mat['data'])
    label = np.array(mat['label'])
    label = label.astype(int)

    I = [[]for i in range(10)]
    for i in range(len(data)):
        I[label[0][i]].append(255-data[i])
    return I

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

        
    
# Calcul des moyennes de chaque chiffre sur l'ensemble des donees dediees au Training
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


def pourcentage(Test,Centroids,N):
    """
    Effectue un Test pour chaque image de l'ensemble 'Test' et verifie le resultat
    Retourne une liste avec les pourcentages d'identification correcte pour chaque chiffre et pour l'ensemble entier
    """
    
    P = [0]*10
    for i in range(10):
        for j in range(len(Test[i])):
            if test(Test[i][j],Centroids,N) == i:
                P[i] += 1
        P[i]/=len(Test[i])
    P.append(sum([P[i]*len(Test[i]) for i in range (10)])/sum([len(Test[i]) for i in range(10)]))
    return P

# Effectue les test pour la norme inf, et les normes-p pour p in [1,20]
def testNorm(Test,Centroids,Nb):
    pourcentages=[]
    #Report_p contient les resultats des pourcentages
    Report_p = []
    Report_p.append(pourcentage(Test,Centroids,np.inf))
    for i in range(1,Nb):
        print('Processing Norm :',i ,"Out of",Nb)
        Report_p.append(pourcentage(Test,Centroids,i))
        
    #plt.bar([i for i in range(Nb)],pourcentages)
    #plt.show()
    return Report_p


#Report dans le terminal des pourcentages par chiffre et par norme
def Report(Report_p,algo):
    N = len(Report_p)
    R = Report_p.copy()

    if algo == 1 :
        for i in range(len(Report_p)):
            for j in range(len(Report_p[i])):
                R[i][j] = '%.5f'%Report_p[i][j]

        print('Percentages of correctly identified digits for each norm')
        print('Norm   |   0   |   1   |   2   |   3   |   4   |   5   |   6   |   7   |   8   |   9   |   total')
        print('inf    ', *R[0] , sep ="|")
        for i in range(1,len(R)):
            print(i,'     ', *R[i] , sep ="|")

            
    elif algo == 2:
        R1 = [R[i][0] for i in range(len(R))]
        R2 = [R[i][1] for i in range(len(R))]
        for i in range(len(R1)):
            print('R1: ',R1[i])
            for j in range(len(R1[i])):
                R1[i][j] = '%.5f'%R1[i][j]
                R2[i][j] = '%.5f'%R2[i][j]

        print('Percentages of correctly identified digits for each k-vector basis')
        print('base size      |   0   |   1   |   2   |   3   |   4   |   5   |   6   |   7   |   8   |   9   |   total')
        for i in range(1,len(R)):
            print(i,'True positifs', *R1[i] , sep ="|")
            print(i,'Rejected     ', *R2[i] , sep ="|")
            
        
    


#SVD
def svd_base(training) :
    bases = [[] for i in range(10)]
    for i in range(10) :
        print('computing base No.',i,'out of 9')
        A = np.matrix(np.vstack([training[i][j] for j in range(len(training[i]))])).transpose()
        bases[i] = np.linalg.svd(A)[0]
    return bases


def calcul_M_k(bases_svd,k):
    """
    calcule pour chaque chiffre à partir de leurs bases SVD la matrice Id-(Uk*Uk^T)
    utilisee pour le calcul des moindres carrés
    """
    print('Number of base vectors:', k)
    bases_k = [bases_svd[i][:,:k] for i in range(10)]
    return [np.identity(28*28)-np.matmul(bases_k[i],bases_k[i].transpose()) for i in range(10)]


def test_svd(image,M_k,threshold) :
    """
    renvoie pour une image le chiffre auquel elle a été identifié ou si le test ne permet pas de conclure 10
    à partir de M_k et avec un euil threshold
    """
    least_squares = [np.linalg.norm(np.matmul(M_k[i],np.array([image]).transpose()),2) for i in range(10)]
    k = np.argmin(least_squares)
    min_1 = least_squares.pop(k)
    min_2 = np.min(least_squares)
    if(min_1 > min_2*threshold):
        return 10
    return k


def pourcentage_SVD(Test,M_k,threshold):
    """
    renvoie le pourcentage de vrais positifs et d'images ecartees pour chaque chiffre et moyen d'une base de donnée Test
    a partir de M_k et avec un seuil threshold
    """
    P1 = [0]*10
    P2 = [0]*10
    for i in range(10):
        print("percentage calculation digit:",i )
        for j in range(len(Test[i])):
            T = test_svd(Test[i][j],M_k,threshold)
            if T == i:
                P1[i] += 1
            elif T == 10:
                P2[i] += 1
        
    P1.append(sum(P1)/sum([len(Test[i])-P2[i] for i in range(10)]))
    P2.append(sum(P2)/sum([len(Test[i]) for i in range(10)]))
    for i in range(10):
        P1[i] /= len(Test[i])-P2[i]
        P2[i] /= len(Test[i])
        
    return P1,P2

def test_bases_SVD(Test,bases,threshold,nb_bases) :
    """
    renvoie la liste des résultats de la fonction pourcentage_SVD en utilisant k bases de la SVD pour k variant de 1 a nb_bases
    """
    report = []
    for k in range(nb_bases) :
        M_k = calcul_M_k(bases,k+1)
        report.append(pourcentage_SVD(Test,M_k,threshold))
    return report
"""
def 
SS = 30
kk = 6


d = 2
M2 = np.zeros((kk-3,SS))
M3 = np.zeros((kk-3,SS))
        
for i in range(3,kk):
    k = i
    V =[d**p for p in range(0,k)]
    print(V)
    for j in range(0,SS):
        S = j
        t0 = time.process_time()
        RechercheExhaustive(k,V,S)
        t1 = time.process_time()
        print(t1-t0)
        M2[i-3,j] = t1-t0
        
        


y = [k for k in range(3,kk)]
x = [k for k in range(0,SS)]
Z = M2

X, Y = np.meshgrid(x, y)
print(Z)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.set_ylabel('K')
ax.set_xlabel('S')
ax.set_zlabel('Temps')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,cmap='viridis',edgecolor='none')
plt.show()f
"""
