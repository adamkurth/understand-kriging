import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import *

nugget = 2.5
sill = 7.5
rang = 10

# assumptions: spherical model, isotropic, 2D

# challenge: how to make it work for 3D?
# challenge: make this accomidate for other model types?
def semivariance(nug, sill, ran, h):
    sv = nug + sill*(3/2*h/ran - 1/2*(h/ran)**3)
    if sv.shape[0] > 1: 
        onescol = np.ones(sv.shape[0])
        sv = np.insert(sv, sv.shape[1], onescol, axis=1)
        onesrow = np.ones(sv.shape[1])
        sv = np.insert(sv, sv.shape[0], onesrow, axis=0)
        sv[sv.shape[0]-1, sv.shape[1]-1] = 0
    else: 
        onescol = np.ones(sv.shape[0])
        sv = np.insert(sv, sv.shape[1], onescol, axis=1)
        sv = sv.T
    return sv

# contour map 

def distance_matrix(X,Y):
    temp_list = []
    # create the distance matrix by traversing through rows and computing each element,
    # then appending the row to the matrix
    for i,j in zip(X,Y):
        for e,d in zip(X,Y):
            dist = sqrt((i-e)**2 + (j-d)**2)
            temp_list.append(dist)
    distance_matrix = np.array([temp_list[x:x+len(X)] for x in range(0, len(temp_list), len(X))])
    return distance_matrix

def distance_to_unknown(X1, Y1, X2, Y2):
    list = []
    # same as above, but for distance from known points to unknown point
    # esentially solving the system of equations for the predicted point
    for k,l in zip(X2, Y2):
        dist = sqrt(((X1-k)**2) + ((Y1-l)**2))
        list.append(dist)
    unknown = np.array(list)
    return unknown

def ok(data_x, data_y, unknown_x, unknown_y, var):
    var_1 = np.reshape(var, (var.shape[0],1)) # column and row value 
    var_1 = var.T
    matrix_distance_known = distance_matrix(data_x, data_y)
    matrix_distance_unknown = distance_to_unknown(unknown_x, unknown_y, data_x, data_y)
    n_sv = semivariance(nugget, sill, rang, matrix_distance_known)

    return None 