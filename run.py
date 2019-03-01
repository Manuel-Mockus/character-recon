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

'''
###############"""""
# Test Algo1
L = centroids(Training)

Afficher(I[2][3])

Report_p = testNorm(I,L,5)

Report(Report_p)
#####################
'''

bases = fn.svd_base(Training)

report = fn.test_bases_SVD(Test,bases,0.95,10)

fn.Report(report,2)




'''
plt.imshow(img.reshape((28,28)),cmap='gray')     
plt.show()
#print(test_svd(test_set[0][0],bases))
'''
