import scipy.io as sio
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
from scipy.signal import convolve2d

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

#renvoie les sets de test (20% de la database, toujours les mêmes) et de training
# (40% de la database choisi aléatoirement d'intersection vide avec le set de test)

def Separation(I):
    Training = [[]for i in range(10)]
    Test = [[]for i in range(10)]
    for i in range(10):
        taille_i = len(I[i])
        listeTraining = [k for k in range(int(taille_i*POURCENT))]
        listeTraining = random.sample(listeTraining,int((taille_i*POURCENT)/2))
        for j in listeTraining:
            Training[i].append(I[i][j])
        for j in range(int(taille_i*POURCENT),taille_i):
            Test[i].append(I[i][j])
    return Training,Test



def Afficher(vect):
    V = vect.reshape((28,28))
    plt.imshow(V, cmap = 'gray')
    plt.axis('off')
    plt.plot() # ne bloque pas la fenetre comme plt.show
    plt.show()


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
    Max_rejected = []
    for i in range(10):
        L = [0]*10
        for j in range(len(Test[i])):
            projection = test(Test[i][j],Centroids,N)
            if projection == i:
                P[i] += 1
            else:
                L[projection]+=1
        chiffre_max = np.argmax(L)
        #print(sum(L)/3500)
        if sum(L) == 0:
            Max_rejected.append(['N/A', 0])
        else:
            Max_rejected.append([chiffre_max, L[chiffre_max]/sum(L)])

        P[i]/=len(Test[i])
    P.append(sum([P[i]*len(Test[i]) for i in range (10)])/sum([len(Test[i]) for i in range(10)]))
    Max_rejected.append(['N/A',0.0])
    return P, Max_rejected

# Effectue les test pour la norme inf, et les normes-p pour p in [1,20]
def testNorm(Test,Centroids,Nb):
    pourcentages=[]
    #Report_p contient les resultats des pourcentages
    Report_p = []
    Report_p.append(pourcentage(Test,Centroids,np.inf))
    for i in range(1,Nb):
        print('Processing Norm :',i ,"Out of",Nb)
        Report_p.append(pourcentage(Test,Centroids,i))

    for i in range(10):
        L = [Report_p[j][0][i] for j in range(len(Report_p))]
        plt.plot([k for k in range(Nb)],L,'--',label = i)

    L = [Report_p[j][0][10] for j in range(len(Report_p))]
    plt.plot([i for i in range(Nb)],L,label = 'total')

    #plt.plot([i for i in range(Nb)],pourcentages)
    plt.xlabel('Norme de Minkowski')
    plt.ylabel('Pourcentage')
    plt.legend()
    plt.show()
    return Report_p


#Report dans le terminal des pourcentages par chiffre et par norme
def Report(Report_p,algo,write = False):
    """
    Si write == True
"""
    if write == True:
        print("Writing report : Report.txt")
        sys.stdout = open("Report.txt",'w')

    N = len(Report_p)
    R = Report_p.copy()
    if algo == 1 :
        for i in range(len(Report_p)):
            for j in range(len(Report_p[i][0])):
                R[i][0][j] = '%.5f'%Report_p[i][0][j]
                R[i][1][j][1] = '%.2f'%Report_p[i][1][j][1]

        print('Percentages of correctly identified digits for each norm')
        print('Norm   |   0   |   1   |   2   |   3   |   4   |   5   |   6   |   7   |   8   |   9   |   total')
        print('inf    ', *R[0][0] , sep ="|")
        print('       ', *R[0][1] , sep ="|")
        for i in range(1,len(R)):
            print(i,'     ', *R[i][0] , sep ="|")
            print('      ', *R[i][1] , sep ="|")



    elif algo == 2:
        R1 = [R[i][0] for i in range(len(R))]
        R2 = [R[i][1] for i in range(len(R))]
        R3 = [R[i][2] for i in range(len(R))]
        for i in range(len(R1)):
            for j in range(len(R1[i])):
                R1[i][j] = '%.5f'%R1[i][j]
                R2[i][j] = '%.5f'%R2[i][j]
                R3[i][j][1] = '%.2f'%R3[i][j][1]

        print('Percentages of correctly identified digits for each k-vector basis')
        print('base size      |   0   |   1   |   2   |   3   |   4   |   5   |   6   |   7   |   8   |   9   |   total')
        for i in range(len(R)):
            print(i+1,'True positifs', *R1[i] , sep ="|")
            print('Rejected      ', *R2[i] , sep ="|")
            print('Mistaken      ', *R3[i] , sep ="|")



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
    à partir de M_k et avec un seuil threshold
    """
    least_squares = [np.linalg.norm(np.matmul(M_k[i],np.array([image]).transpose()),2) for i in range(10)]
    k = np.argmin(least_squares)
    min_1 = least_squares.pop(k)
    min_2 = np.min(least_squares)
    if(min_1 > min_2*threshold):
        return 10
    return k


def affiche_norme_SVD(bases,k,Test,i):
    """
    affiche la norme du résultat du calcul matriciel obtenu lors du calcul de la SVD pour chaque
    exemplaire du chiffre i par rapport aux bases des différents chiffres
    """
    M_k = calcul_M_k(bases,k)
    abs = [i for i in range(10)]
    for j in range(len(Test[i])):
        least_squares = [np.linalg.norm(np.matmul(M_k[l],np.array([Test[i][j]]).transpose()),2) for l in range(10)]
        plt.plot(abs,least_squares)
        plt.xlabel('Base')
        plt.ylabel('Distance à la base')
        plt.xticks([i for i in range(10)],[i for i in range(10)])
        plt.legend()
    plt.show()

def pourcentage_SVD(Test,M_k,threshold):
    """
    renvoie le pourcentage de vrais positifs et d'images ecartees pour chaque chiffre et moyen d'une base de donnée Test
    a partir de M_k et avec un seuil threshold
    """
    P1 = [0]*10 # pourcentage vrai positifs
    P2 = [0]*10 # pourcentage rejetés
    P3 = [] # (chiffre le plus confondu, pourcentage)
    for i in range(10):
        L= [0]*10
        print("processing digit", i)
        for j in range(len(Test[i])):
            T = test_svd(Test[i][j],M_k,threshold)
            if T == i:
                P1[i] += 1
            elif T == 10:
                P2[i] += 1
            else:
                L[T] += 1

        chiffre_max = np.argmax(L)
        if sum(L) == 0:
            P3.append(['N/A', 0])
        else:
            P3.append([chiffre_max, L[chiffre_max]/sum(L)])
    P3.append(['N/A', 0])


    if sum([len(Test[i])-P2[i] for i in range(10)]) == 0:
        #si tous les chiffres ont ete rejetes par l'algorithme
        P1.append(1)
    else:
        P1.append(sum(P1)/sum([len(Test[i])-P2[i] for i in range(10)]))

    P2.append(sum(P2)/sum([len(Test[i]) for i in range(10)]))
    for i in range(10):
        if len(Test[i])-P2[i] == 0:
            P1[i] = 1
        else:
            P1[i] /= len(Test[i])-P2[i]

        P2[i] /= len(Test[i])

    return P1,P2,P3

def test_bases_SVD(Test,bases,threshold,nb_bases) :
    """
    renvoie la liste des résultats de la fonction pourcentage_SVD en utilisant k bases de la SVD pour k variant de 1 a nb_bases
    """
    report = []
    for k in range(nb_bases) :
        M_k = calcul_M_k(bases,k+1)
        report.append(pourcentage_SVD(Test,M_k,threshold))


    for i in range(10):
        L = [report[j][0][i] for j in range(len(report))]
        plt.plot([1+k for k in range(len(report))],L,'--',label = i)

    L = [report[j][0][10] for j in range(len(report))]
    plt.plot([i+1 for i in range(len(report))],L,label = 'total')

    #plt.plot([i for i in range(Nb)],pourcentages)
    plt.xlabel('Nombre de vecteurs de base')
    plt.ylabel('Pourcentage')
    plt.legend()
    plt.show()

    return report

def SVD_show_3D(Test,bases,nb_t,min_k,max_k):

    threshold_min = 0.96 #seuil minimal
    thresholds = np.linspace(threshold_min,1,nb_t)
    Z1 = np.zeros((max_k-min_k+1,nb_t))
    Z2 = np.zeros((max_k-min_k+1,nb_t))
    report = []
    for k in range(min_k,max_k+1) :
        print("test with ",k+1,"basis vectors")
        M_k = calcul_M_k(bases,k+1)
        for j in range(nb_t):
            print("treshold : ", thresholds[j])
            P1,P2 = pourcentage_SVD(Test,M_k,thresholds[j])
            Z1[k-min_k,j] = P1[10]
            Z2[k-min_k,j] = P2[10]
            report.append((P1,P2))

    y = [k+1 for k in range(min_k,max_k+1)]
    x = thresholds

    X, Y = np.meshgrid(x, y)

    fig1 = plt.figure(1)
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z1, 50, cmap='binary')
    ax.set_ylabel('basis vectors')
    ax.set_xlabel('threshold')
    ax.set_zlabel('True positive percentage')
    ax.plot_surface(X, Y, Z1, rstride=1, cstride=1,cmap='viridis',edgecolor='none')

    fig2 = plt.figure(2)
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z2, 50, cmap='binary')
    ax.set_ylabel('basis vectors')
    ax.set_xlabel('threshold')
    ax.set_zlabel('Rejected percentage')
    ax.plot_surface(X, Y, Z2, rstride=1, cstride=1,cmap='viridis',edgecolor='none')
    plt.show()
    return report



def SVD_show_2D(Test,bases,nb_t,min_t,max_t):

    thresholds = np.linspace(min_t,max_t,nb_t,endpoint = True)
    Z1 = np.zeros(nb_t)
    Z2 = np.zeros(nb_t)
    report = []
    M_k = calcul_M_k(bases,10)
    for j in range(nb_t):
        print("treshold : ", thresholds[j])
        P1,P2 = pourcentage_SVD(Test,M_k,thresholds[j])
        Z1[j] = P1[10]
        Z2[j] = P2[10]
        report.append((P1,P2))

    X = thresholds

    fig1 = plt.figure(1)
    plt.plot(X,Z1,'-x')
    plt.ylabel('True positives percentage')
    plt.xlabel('threshold')


    fig2 = plt.figure(2)
    plt.plot(X,Z2,'-o')
    plt.ylabel('Rejected percentage')
    plt.xlabel('threshold')

    plt.show()
    return report



##########################
#####TANGENT DISTANCE#####
##########################

def Translation(img,k,axis = 1):
    #axis  1 = x, 0 = y
    img1 = np.copy(img)
    img1 = img1.reshape(28,28)
    img1 = np.roll(img1, k, axis) # translation en X
    img1 = img1.reshape(784)
    return img1

def diff_x(img):
    I = 255 - img.reshape(28,28)
    filtre_x = np.array([[0,0,0],[-1,1,0],[0,0,0]])
    Ix = convolve2d(I,filtre_x)
    Ix = Ix[1:-1,1:-1]
    return Ix.reshape(784)

def diff_y(img):
    I = 255 - img.reshape(28,28)
    filtre_y = np.array([[0,-1,0],[0,1,0],[0,0,0]])
    Iy = convolve2d(I,filtre_y)
    Iy = Iy[1:-1,1:-1]
    return Iy.reshape(784)

def diff_rotate(img):
     px = diff_x(img).reshape(28,28)
     py = diff_y(img).reshape(28,28)
     coords = np.matrix([i for i in range(-14,14)]) # pour que la rotation soit par rapport au centre
     ones = np.matrix(np.ones(28))
     y = ones.transpose() * coords
     x = coords.transpose() * ones
     D_rotation = np.multiply(y,px) - np.multiply(x,py)
     return D_rotation.reshape(784)

def diff_scaling(img):
    px = diff_x(img).reshape(28,28)
    py = diff_y(img).reshape(28,28)
    coords = np.matrix([i for i in range(-14,14)]) # pour que le scaling soit par rapport au centre :)
    ones = np.matrix(np.ones(28))
    y = ones.transpose() * coords
    x = coords.transpose() * ones
    D_scaling = np.multiply(x,px) + np.multiply(y,py)
    return D_scaling.reshape(784)

def diff_thickening(img):
    px = diff_x(img).reshape(28,28)
    py = diff_y(img).reshape(28,28)
    D_scaling = np.multiply(px,px) + np.multiply(py,py)
    return D_scaling.reshape(784)

def diff_PHT(img):
    px = diff_x(img).reshape(28,28)
    py = diff_y(img).reshape(28,28)
    coords = np.matrix([i for i in range(-14,14)])
    ones = np.matrix(np.ones(28))
    y = ones.transpose() * coords
    x = coords.transpose() * ones
    D_scaling = np.multiply(x,px) - np.multiply(y,py)
    return D_scaling.reshape(784)

def diff_DHT(img):
    px = diff_x(img).reshape(28,28)
    py = diff_y(img).reshape(28,28)
    coords = np.matrix([i for i in range(-14,14)])
    ones = np.matrix(np.ones(28))
    y = ones.transpose() * coords
    x = coords.transpose() * ones
    D_scaling = np.multiply(y,px) + np.multiply(x,py)
    return D_scaling.reshape(784)


def find_min(p,Te,e,func):
    Tp = func(p)
    Tp = np.matrix(Tp).transpose()
    A = np.hstack([-Tp,Te])
    U,S1,V = np.linalg.svd(A)
    b = np.transpose([p-e])
    S = np.matrix(np.diag(S1))
    """
    print("V",V.shape)
    print("U",U.shape)
    print("S",S.shape)
    """
    x = V.transpose()*np.linalg.solve(S,U[:,:2].transpose()*b)
    """
    print(x)
    print(np.linalg.norm(A*x - b))
    print(np.linalg.norm(A*np.matrix([[1],[1]]) - b))
    print(A)

    print(np.linalg.norm(A*x - b))
    """

    return np.linalg.norm(A*x - b)

def find_min2(p,Te,e,funcs):
    A = Te
    l = len(funcs)
    for k in range(l) :
        Tp = funcs[k](p)
        Tp = np.matrix(Tp).transpose()
        A = np.hstack([-Tp,A])
    U,S1,V = np.linalg.svd(A)
    b = np.transpose([p-e])
    S = np.matrix(np.diag(S1))
    """
    print("V",V.shape)
    print("U",U.shape)
    print("S",S.shape)
"""
    x = V.transpose()*np.linalg.solve(S,U[:,:2*l].transpose()*b)
    """
    print(x)
    print(np.linalg.norm(A*x - b))
    print(np.linalg.norm(A*np.matrix([[1],[1]]) - b))
    prinhttp://www.np.hstack/t(A)

    print(np.linalg.norm(A*x - b))
    """

    return np.linalg.norm(A*x - b)

def TTT(Centroids,Test,func):
    Te = []
    P = [0]*10 # pourcentage vrai positifs
    for i in range(10):
        T = np.matrix(func(Centroids[i]))
        T = T.transpose()
        Te.append(T)
    for i in range(10):
        print("chiffre ",i)
        for j in range(len(Test[i])):
            x = [0]*10
            for k in range(10):
                x[k]= find_min(Test[i][j],Te[k],Centroids[k],func)
            T = np.argmin(x)
            #print(T,i)
            if T == i:
                P[i] += 1

    P.append(sum(P)/sum([len(Test[i]) for i in range(10)]))
    for i in range(10) :
        P[i] /= len(Test[i])
    return P

def TTT2(Centroids,Test,funcs):
    Te = []
    P = [0]*10 # pourcentage vrai positifs
    for i in range(10):
        TT = np.array([]).reshape(28*28,0)
        for j in range(len(funcs)) :
            T = np.matrix(funcs[j](Centroids[i]))
            T = T.transpose()
            TT = np.hstack([T,TT])
        Te.append(TT)
    for i in range(10):
        print("chiffre ",i)
        for j in range(len(Test[i])):
            x = [0]*10
            for k in range(10):
                x[k]= find_min2(Test[i][j],Te[k],Centroids[k],funcs)
            T = np.argmin(x)
            #print(T,i)
            if T == i:
                P[i] += 1

    P.append(sum(P)/sum([len(Test[i]) for i in range(10)]))
    for i in range(10) :
        P[i] /= len(Test[i])
    return P
