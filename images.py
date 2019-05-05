import scipy.io as sio
import numpy as np
import random
import matplotlib.pyplot as plt
import sys
import functions as fn
#lecture Base de donees

filename = 'mnist-original.mat'
I = fn.read_database(filename)

Training,Test = fn.Separation(I)

Centroids = fn.centroids(Training)
"""
img = Test[5][411]
fn.AfficherBord(img)
der = fn.diff_x(img)
img = fn.Translation(img,3)
fn.AfficherBord(img)
fn.AfficherBord(der)

img = Test[2][66]
fn.AfficherBord(img)
der = fn.diff_rotate(img)
img = fn.Rotation(img,20)
fn.AfficherBord(img)
fn.AfficherBord(der)

img = Test[6][10]
fn.AfficherBord(img)
der = fn.diff_scaling(img)
img = fn.Scaling(img,1.2)
fn.AfficherBord(img)
fn.AfficherBord(der)


img = Test[9][14]
fn.AfficherBord(img)
der = fn.diff_PHT(img)
img = fn.PHT(img,1.2)
fn.AfficherBord(img)
fn.AfficherBord(der)

img = Test[3][7]
fn.AfficherBord(img)
der = fn.diff_DHT(img)
img = fn.DHT(img,0.2)
fn.AfficherBord(img)
fn.AfficherBord(der)
"""
img = Test[4][15]
fn.AfficherBord(img)
der = fn.diff_thickening(img)
img = fn.Thickening(img,thicken = False)
fn.AfficherBord(img)
fn.AfficherBord(der)
