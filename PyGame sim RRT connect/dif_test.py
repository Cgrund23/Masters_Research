import random
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import cholesky, solve, solve_triangular
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# Define the RBF kernel function
def rbf_kernel(X1, X2, length_scale, sigma_f):
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * X1 @ X2.T
    return sigma_f * np.exp(-0.5 * (sqdist / length_scale**2))

# Define the CBF function
def CBF(x, y):
    points = np.array([x, y]).T
    length_scale = 10
    sigma_f = 1

    x_width = 500
    y_width = 500
    resolution = 2
    grid_size = int(x_width * 1 / resolution)
    x_grid, y_grid = np.meshgrid(np.linspace(0, x_width, grid_size), np.linspace(0, y_width, grid_size))
    safety_matrix = np.column_stack((x_grid.ravel(), y_grid.ravel()))

    NY = np.ones((len(points), 1))
    X_query = np.array([200, 200]).reshape((1, 2))

    k_star = rbf_kernel(points, safety_matrix, length_scale, sigma_f)
    K = rbf_kernel(points, points, length_scale, sigma_f)
    K_self = rbf_kernel(points, X_query, length_scale, sigma_f)
    k_inv = np.linalg.pinv(K)

    return (1 - 2 * (k_star.T @ k_inv @ NY)), grid_size, x_grid, y_grid

def CBF_derivative(x, y):
    points = np.array([x, y]).T
    length_scale = 10
    sigma_f = 1

    X_query = np.array([250, 250]).reshape((1, 2))

    x_width = 500
    y_width = 500
    resolution = 2
    grid_size = int(x_width * 1 / resolution)
    x_grid, y_grid = np.meshgrid(np.linspace(0, x_width, grid_size), np.linspace(0, y_width, grid_size))
    safety_matrix = np.column_stack((x_grid.ravel(), y_grid.ravel()))

    NY = 1 * np.ones((len(points), 1))
    k_star = rbf_kernel(points, safety_matrix, length_scale, sigma_f)
    #k_star = rbf_kernel(points, X_query, length_scale, sigma_f)
    K = rbf_kernel(points, points, length_scale, sigma_f)
    k_inv = np.linalg.pinv(K)

    # # Compute the pairwise differences for x and y coordinates
    diff_x = safety_matrix[:, 0].reshape(-1, 1) - points[:, 0].reshape(1, -1)
    diff_y = safety_matrix[:, 1].reshape(-1, 1) - points[:, 1].reshape(1, -1)
    
    # Compute the pairwise differences for x and y coordinates
    # diff_x = X_query[:, 0].reshape(-1, 1) - points[:, 0].reshape(1, -1)
    # diff_y = X_query[:, 1].reshape(-1, 1) - points[:, 1].reshape(1, -1)

    # Compute the derivatives of the RBF kernel with respect to x and y
    grad_k_star_x = -(diff_x / (length_scale**2)) * k_star.T
    grad_k_star_y = -(diff_y / (length_scale**2)) * k_star.T

    # Compute the derivative of the CBF function
    grad_h_x = -2 * (grad_k_star_x @ k_inv @ NY).flatten()
    grad_h_y = -2 * (grad_k_star_y @ k_inv @ NY).flatten()

    # Reshape the gradients to the grid size
    grad_h_x = grad_h_x.reshape(grid_size, grid_size)
    grad_h_y = grad_h_y.reshape(grid_size, grid_size)
    
    gradient_magnitude = np.sqrt(grad_h_x**2 + grad_h_y**2)

    return gradient_magnitude

def dCBF(x,y):
    points = np.array([x, y]).T
    length_scale = 10
    sigma_f = 1

    x_width = 500
    y_width = 500
    resolution = 2
    grid_size = int(x_width * 1 / resolution)
    x_grid, y_grid = np.meshgrid(np.linspace(0, x_width, grid_size), np.linspace(0, y_width, grid_size))
    safety_matrix = np.column_stack((x_grid.ravel(), y_grid.ravel()))

    NY = -1 * np.ones((len(points), 1))
    Y = np.ones((len(points), 1))
    X_query = np.array([0, 0]).reshape((1, 2))

    k_star = rbf_kernel(points, safety_matrix, length_scale, sigma_f)
    K = rbf_kernel(points, points, length_scale, sigma_f)
    K_self = rbf_kernel(X_query, points, length_scale, sigma_f)
    k_inv = np.linalg.pinv(K)

    dcbf = np.zeros((2, 1), dtype=float)

    return dh_dx

# Create the gradient magnitude heat map
arr = np.zeros((500, 500), dtype=np.float32)
imgsize = arr.shape[:2]

# Define the center and target radius
center_x, center_y = imgsize[0] // 2, imgsize[1] // 2
target_radius = 50

# Initialize the gradient magnitude array
gradient_magnitude = np.zeros(imgsize)

for y in range(imgsize[1]):
    for x in range(imgsize[0]):
        dx = x - center_x
        dy = y - center_y
        distanceToCenter = np.sqrt(dx**2 + dy**2)

        # Only calculate the gradient outside the circle
        if distanceToCenter > target_radius:
            grad_x = dx / target_radius
            grad_y = dy / target_radius
            gradient_magnitude[y, x] = np.sqrt(grad_x**2 + grad_y**2)
        else:
            gradient_magnitude[y, x] = 0

# Generate points for CBF
num_samples = 10
theta = np.linspace(0, 2 * np.pi, num_samples)
r = target_radius
x, y = r * np.cos(theta) + center_x, r * np.sin(theta) + center_y
h_world, grid_size, x_grid, y_grid = CBF(x, y)
dcbf = CBF_derivative(x,y)
h_grid = h_world.reshape(grid_size, grid_size)
d_grid = dcbf.reshape(grid_size,grid_size)
# Plot the first figure (Gradient Magnitude Heat Map)
fig1, ax1 = plt.subplots()
contour = ax1.contourf(x_grid, y_grid, h_grid, levels=20, cmap='twilight')
cb2 = plt.colorbar(contour, ax=ax1, orientation='vertical', label='CBF Value')
# heatmap1 = ax1.imshow(gradient_magnitude, cmap='plasma')
# cb1 = plt.colorbar(heatmap1, ax=ax1, orientation='vertical', label='Gradient Magnitude')

# Draw the circle representing the target radius and points
draw_circle1 = plt.Circle((center_x, center_y), target_radius, color='white', fill=False, linewidth=2)
ax1.add_patch(draw_circle1)
ax1.scatter(x, y, marker='o', linewidth=1, color='black', label='Samples')

# Add titles and labels
ax1.set_title("DCBF Contour Heat Map")
ax1.set_xlabel("X-axis")
ax1.set_ylabel("Y-axis")

# Plot the second figure (CBF Heat Map)
fig2, ax2 = plt.subplots()
contour = ax2.contourf(x_grid, y_grid, d_grid, levels=20, cmap='twilight')
cb2 = plt.colorbar(contour, ax=ax2, orientation='vertical', label='CBF Value')

# Plot the sampled points and reference circle
ax2.scatter(x, y, marker='o', linewidth=1, color='black', label='Samples')
draw_circle2 = plt.Circle((center_x, center_y), target_radius, color='white', fill=False, linewidth=2)
ax2.add_patch(draw_circle2)

# Add titles and labels
ax2.set_title("DCBF Contour Heat Map")
ax2.set_xlabel("X-axis")
ax2.set_ylabel("Y-axis")

# Display both figures
plt.show()