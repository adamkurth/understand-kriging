import os
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import GPy

def load_file_h5(filename, dataset_path):
    if not os.path.exists(filename):
        print("File not found:", filename)
        return None
    try:
        with h5.File(filename, "r") as f:
            data = np.array(f[dataset_path])
            print("Loaded data successfully from", filename)
            return data
    except Exception as e:
        print("An error occurred:", e)
        return None

def visualize_initial_data(X, Y, Z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X, Y, Z, c=Z, cmap='viridis')
    plt.colorbar(scatter, label='Intensity')
    ax.set_title('Initial Data')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Value')
    plt.show()

def visualize_gp_fit(X, Y, Z, gp):
    y_pred, variance = gp.predict(np.column_stack([X, Y]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X, Y, y_pred, c=y_pred, cmap='viridis')
    plt.colorbar(scatter, label='Predicted Intensity')
    ax.set_title('Gaussian Process Fit')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Predicted Z Value')
    plt.show()

def create_3d_scatter(x, y, z, title='3D Scatter Plot'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x, y, z, c=z, cmap='viridis')
    cbar = plt.colorbar(scatter)
    cbar.set_label('Intensity')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Value')
    plt.title(title)
    plt.show()

def create_2d_scatter(x, y, title='2D Scatter Plot'):
    plt.figure()
    plt.scatter(x, y, c='b', marker='o')
    plt.title(title)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()

def main():
    # Load data
    data = load_file_h5("9_18_23_high_intensity_3e8keV-1.h5", "/entry/data/data")
    if data is None:
        print("Data could not be loaded. Please check the file path and dataset name.")
        return

    # Downsample data for faster processing
    downsample_factor = 10
    data = data[::downsample_factor, ::downsample_factor]

    # Prepare coordinates (X, Y) and observed values (Z)
    X, Y = np.mgrid[0:data.shape[0], 0:data.shape[1]]
    X = X.flatten()
    Y = Y.flatten()
    Z = data.flatten()

    # Visualize initial data
    # visualize_initial_data(X, Y, Z)

    # Define a sparse GP model using GPy
    kernel = GPy.kern.RBF(input_dim=2, variance=1., lengthscale=1.)
    m = GPy.models.SparseGPRegression(np.column_stack([X, Y]), Z[:, None], kernel, num_inducing=500)

    # Optimize the model
    m.optimize('bfgs')

    # Visualize GP Fit
    visualize_gp_fit(X, Y, Z, m)

    # Make predictions
    X_pred = np.column_stack([np.linspace(0, data.shape[0], 1000), np.linspace(0, data.shape[1], 1000)])
    y_pred, variance = m.predict(X_pred)

    # Visualize Results
    create_3d_scatter(X, Y, Z, '3D Scatter Plot of Original Data')
    create_3d_scatter(X_pred[:, 0], X_pred[:, 1], y_pred, '3D Scatter Plot with Gaussian Process Predictions')
    create_2d_scatter(X_pred[:, 0], X_pred[:, 1], '2D Scatter Plot with Gaussian Process Predictions')

if __name__ == "__main__":
    main()
