import scipy.io as sio
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt

mat = sio.loadmat('dataset_Octave.mat')
data = np.transpose(mat['data'])
label = np.array(mat['label'])
label = label.astype(int)

I = [[]for i in range(10)]
print(len(data))
for i in range(len(data)):
    I[label[0][i]].append(255-data[i])

K = 301
img = Image.new('L',(28,28))
img.putdata(I[2][K])
nimg = img.resize((280,280))
nimg.show()
for i in range(len(I)):
    print(len(I[i]))


def centroids(Training):
    L = [np.zeros(28*28) for i in range(10)]
    for i in range(10):
        for j in range(len(Training[i])):
            L[i] = np.add(L[i],Training[i][j])
        L[i] = L[i]/len(Training[i])
        L[i] = L[i].astype(int) 
    return L

L = centroids(I)

img = Image.new('L',(28,28))
img.putdata(L[1])
nimg = img.resize((280,280))
nimg.show()

def test(Image,Centroids,N):
    return np.argmin([np.linalg.norm(Centroids[i]-Image,N) for i in range(10)]) 


def pourcentage(Tab,Centroids,N):
    P = [0]*10
    for i in range(10):
        for j in range(len(Tab[i])):
            if test(Tab[i][j],Centroids,N) == i:
                P[i] += 1
        P[i]/=len(Tab[i])
    return P

                
print(test(I[2][K],L,2))
P = pourcentage(I,L,2)
print(P)
print(np.mean(P))
