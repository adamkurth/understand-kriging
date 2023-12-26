import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt

# Parameters for kriging
nugget = 2.5
sill = 7.5
rang = 10

# Data description:
# x: Easting (meters) in RDH (Netherlands topographical) map coordinates
# y: Northing (meters) in RDH map coordinates
# zinc: Soil zinc (ppm)

"""Calculate semi-variance matrix"""
def semivariance(nug, sill, ran, h):
    sv = nug + sill * (1.5 * h / ran - 0.5 * (h / ran) ** 3)
    sv[h > ran] = nug + sill
    return sv

"""Calculate distance matrix between known points"""
def distance_matrix(X, Y):
    n = len(X)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = sqrt((X[i] - X[j]) ** 2 + (Y[i] - Y[j]) ** 2)
    return dist_matrix

"""Ordinary kriging interpolation for a single point"""
def ordinary_kriging(data_x, data_y, unknown_x, unknown_y, var, nugget, sill, rang):
    matrix_distance_known = distance_matrix(data_x, data_y)
    matrix_distance_unknown = np.array([[sqrt((unknown_x - data_x[i]) ** 2 + (unknown_y - data_y[i]) ** 2) for i in range(len(data_x))]])
    known_sv = semivariance(nugget, sill, rang, matrix_distance_known)
    unknown_sv = semivariance(nugget, sill, rang, matrix_distance_unknown)

    known_sv_inv = np.linalg.inv(known_sv)
    weights = np.matmul(known_sv_inv, unknown_sv.T)
    est = np.dot(weights.T, var)
    return est[0]

"""Interpolate the data using ordinary kriging"""
def interpolation(X, Y, variable, res):
    resolution = res
    X_mesh, Y_mesh = np.meshgrid(
        np.linspace(np.amin(X) - 1, np.amax(X) + 1, resolution),
        np.linspace(np.amin(Y) - 1, np.amax(Y) + 1, resolution),
    )
    estimated_values = np.zeros((resolution * resolution, 1))

    for i in range(resolution * resolution):
        x, y = X_mesh.flat[i], Y_mesh.flat[i]
        estimated = ordinary_kriging(X, Y, x, y, variable, nugget, sill, rang)
        estimated_values[i] = estimated

    grid = pd.DataFrame(data={"X": X_mesh.ravel(), "Y": Y_mesh.ravel(), "Z": estimated_values.ravel()})
    return grid

"""Load the Meuse dataset"""
def load_meuse():
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    
    pandas2ri.activate()
    sp = importr('sp')
    ro.r('data(meuse)')  # Load the Meuse dataset into R's environment
    meuse_df = pandas2ri.rpy2py(ro.r['meuse'])
    return meuse_df

"""Check if a matrix is square"""
def is_square(matrix):
    if matrix.ndim < 2:
        return False
    return matrix.shape[0] == matrix.shape[1]

# Load Meuse dataset
data = load_meuse() 
X = data['x'].values
Y = data['y'].values
variable = data['zinc'].values

# Perform kriging interpolation and plot results
for res in [10, 20, 50]:
    grid = interpolation(X, Y, variable, res)
    plt.figure(figsize=(8, 6))
    plt.contourf(
        grid["X"].values.reshape(res, res),
        grid["Y"].values.reshape(res, res),
        grid["Z"].values.reshape(res, res),
        cmap='viridis',
        levels=15,
    )
    plt.colorbar(label='Interpolated Value')
plt.show()

