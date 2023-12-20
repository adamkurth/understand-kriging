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
""" Calculate semi-variance matrix """
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

# contour map ??

"""Calculate distance to known points"""
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

"""Calculate distance to unknown points"""
def distance_to_unknown(X1, Y1, X2, Y2):
    list = []
    # same as above, but for distance from known points to unknown point
    # esentially solving the system of equations for the predicted point
    for k,l in zip(X2, Y2):
        dist = sqrt(((X1-k)**2) + ((Y1-l)**2))
        list.append(dist)
    unknown = np.array(list)
    return unknown

"""Returns best estimate of unknown point based on known points and their values"""
def ordinary_kriging(data_x, data_y, unknown_x, unknown_y, var):
    # reshape the variance array to be a column vector
    var_1 = np.reshape(var, (var.shape[0],1)) # column and row value 
    var_1 = var.T
    # distance matrix of known and unknown points
    matrix_distance_known = distance_matrix(data_x, data_y)
    matrix_distance_unknown = distance_to_unknown(unknown_x, unknown_y, data_x, data_y)
    # known and unknown semi-variance matrix
    known_sv = semivariance(nugget, sill, rang, matrix_distance_known)
    unknown_sv = semivariance(nugget, sill, rang, matrix_distance_unknown)
    # inverse of known semi-variance matrix
    known_sv_inv = np.linalg.inv(known_sv)
    # weights matrix (semi-variance matrix * column)
    weights = np.matmul(known_sv_inv, unknown_sv)
    # this ensures the dot product works properly
    weights = np.delete(weights, weights.shape[0]-1, axis=0) # delete the last row 
    est = np.dot(var_1, weights) # estimated value
    return est[0] 

""" Interpolate the data utilize ordinary kriging"""
def interpolation(X,Y, variable, res):
    # create a meshgrid of the points to be interpolated
    resolution = res # resolution of the interpolation
    X_mesh = np.linspace(np.amin(X)-1, np.amax(X)+1, resolution)
    Y_mesh = np.linspace(np.amin(Y)-1, np.amax(Y)+1, resolution)
    XX, YY = np.meshgrid(X_mesh, Y_mesh)
    EX = []
    EY = []
    EZ = []
    
    for x in np.nditer(XX):
        EX.append(float(x))
    for y in np.nditer(YY):
        EY.append(float(y))
    grid_1 = pd.DataFrame(data={'X':EX, 'Y':EY})
    for i, row in grid_1.iterrows():
        estimated = ordinary_kriging(X, Y, row['X'], row['Y'], variable)
        EZ.append(estimated)
    grid = pd.DataFrame(data={'X':EX, 'Y':EY, 'Z':EZ})
    return grid

# OBTAIN DATA

data = pd.read_csv('data.csv') 
X = data['X'].to_numpy()
Y = data['Y'].to_numpy()
var = data['var'].to_numpy()

test = interpolation(X, Y, var)
print(test)