import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky, solve, solve_triangular

# function for RBF kernel
def rbf_kernel(X1, X2, length_scale, sigma_f):
    """
    Computes the RBF (Radial Basis Function) kernel between X1 and X2.
    """
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * X1 @ X2.T  # distance between points in X1 and X2
                                                                                 # note the dimentions in the sums!
                                                                                 # This is to create a matrix containing 
                                                                                 # all distances between pairs of points
    return sigma_f**2 * np.exp(-0.5 * sqdist / length_scale**2)  # Same kernel as in paper

# function defining CBF (simple - not the one on GP Lidar paper)
def cbf_function(x_test, safe_dist):
    """
    Computes the CBF at certain distance.
    """
    return  rbf_kernel(x_test, X_train, length_scale, sigma_f) @ alpha - safe_dist



# Step 1: Simulate LiDAR Data
num_points = 20
x_width = 10
y_width = 10
x0, y0 = 0, 0

num_lines = num_points
x_start = -6
x_end = 6
y_line = 6

angles = np.linspace(0, np.pi, num_lines)

line_start = np.array([x_start, y_line])
line_end = np.array([x_end, y_line])

x_lidar = []
y_lidar = []

for theta in angles:
    m = np.tan(theta)
    if m != 0:
        x_intersect = y_line / m
        if x_start <= x_intersect <= x_end:
            x_lidar.append(x_intersect)
            y_lidar.append(y_line)

x_lidar = np.array(x_lidar)
y_lidar = np.array(y_lidar)

distances = np.sqrt(x_lidar**2 + y_lidar**2)
angles = np.arctan2(y_lidar, x_lidar)

plt.figure(1)
plt.scatter(x0, y0, color='red', label='Sensor Position')
plt.scatter(x_lidar, y_lidar, color='blue', label='LiDAR Points')
plt.title('Simulated LiDAR Data')
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.xlim([-x_width, x_width])
plt.ylim([-y_width, y_width])
plt.grid(True)
plt.legend()


# Step 2: Gaussian Process (GP) Model Training
sigma_f = 1.0  # set to 1 as in paper
length_scale = 1.0
noise_variance = 1e-4

X_train = np.column_stack((x_lidar, y_lidar))
Y_train = distances

# Compute the covariance matrices
K = rbf_kernel(X_train, X_train, length_scale, sigma_f) + noise_variance * np.eye(len(X_train))
alpha = np.linalg.pinv(K) @ Y_train

# Define a grid for visualization
grid_size = 100
x_grid, y_grid = np.meshgrid(np.linspace(-x_width, x_width, grid_size), np.linspace(-y_width, y_width, grid_size))
X_test = np.column_stack((x_grid.ravel(), y_grid.ravel()))
print(X_test)
# Predict GP mean and variance
K_star = rbf_kernel(X_test, X_train, length_scale, sigma_f)
K_ss = rbf_kernel(X_test, X_test, length_scale, sigma_f) + noise_variance * np.eye(len(X_test))

mu_test = K_star @ alpha
var_test = np.diag(K_ss - K_star @ np.linalg.pinv(K) @ K_star.T)

# Reshape for plotting
mu_grid = mu_test.reshape(grid_size, grid_size)
var_grid = var_test.reshape(grid_size, grid_size)

plt.figure(2)
plt.contourf(x_grid, y_grid, mu_grid, 20, cmap='viridis')
plt.colorbar()
plt.title('GP Mean (Gaussian Kernels)')
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.scatter(x_lidar, y_lidar, color='red', label='LiDAR Points')
plt.xlim([-x_width, x_width])
plt.ylim([-y_width, y_width])
plt.legend()


plt.figure(3)
plt.contourf(x_grid, y_grid, var_grid, 20, cmap='viridis')
plt.colorbar()
plt.title('GP Variance (Uncertainty)')
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.scatter(x_lidar, y_lidar, color='red', label='LiDAR Points')
plt.xlim([-x_width, x_width])
plt.ylim([-y_width, y_width])
plt.legend()


# Step 4: Define and Visualize trivial Control Barrier Function (CBF) 
safe_distance = 0.5

cbf_values = cbf_function(X_test, safe_distance)
cbf_grid = cbf_values.reshape(grid_size, grid_size)

plt.figure(4)
plt.contourf(x_grid, y_grid, cbf_grid, 20, cmap='viridis')
plt.colorbar()
plt.title('Control Barrier Function (CBF)')
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.scatter(x_lidar, y_lidar, color='red', label='LiDAR Points')
# Plotting contour at h(x) = 0
plt.contour(x_grid, y_grid, cbf_grid, [0], colors='black', linewidths=2)  # CBF boundary
plt.xlim([-x_width, x_width])
plt.ylim([-y_width, y_width])
plt.legend()

plt.show()
