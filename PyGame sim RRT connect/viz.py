import random
import time
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import numpy as np
from scipy.linalg import cholesky, solve, solve_triangular
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from shapely.geometry import Point, Polygon as ShapelyPolygon


class MyFig:
    def __init__(self,grid_size=12,) -> None:

        plt.ion()
        self.fig, self.ax = plt.subplots()
        grid = grid_size**2*2
        x_width = grid_size
        y_width = grid_size
        self.x_grid, self.y_grid = np.meshgrid(np.linspace(-x_width, x_width, grid), np.linspace(-y_width, y_width, grid))
        X_test = np.column_stack((self.x_grid.ravel(), self.y_grid.ravel()))
        self.Z = np.sin(self.x_grid) * np.cos(self.y_grid)
        self.ax.set_title('CBF vs Barrier')
        self.contour = self.ax.contourf(self.x_grid, self.y_grid, self.Z, cmap='plasma')
        self.colorbar = self.fig.colorbar(self.contour)
        pass



    def sample_points_along_polygons(self, pgons, num_samples=10):
        """
        Returns a list of sampled points along the exterior of multiple polygons.

        pgons: List of Matplotlib Polygon objects or arrays of vertices.
        num_samples: Number of points to sample along each polygon's exterior.
        """
        sampled_coords = []
        distances_all = []

        for pgon in pgons:
            # Extract vertices using .get_path().vertices (works for both Polygon and CirclePolygon)
            vertices = pgon.get_path().vertices
            shapely_pgon = ShapelyPolygon(vertices)

            # Sample points along the exterior of the polygon
            exterior = shapely_pgon.exterior
            distances = np.linspace(0, exterior.length, num_samples)
            sampled_points = [exterior.interpolate(dist) for dist in distances]

            # Collect coordinates of each sampled point
            sampled_coords.extend([(pt.x, pt.y) for pt in sampled_points])
        sampled_coords_all = np.array(sampled_coords).reshape((-1,2))

        return distances_all, sampled_coords_all
    # Define the CBF function

    def CBF(self, pgons):
        dist, points = self.sample_points_along_polygons(pgons)
        length_scale = 0.7
        sigma_f = 1

        x_width = 21
        y_width = 21
        resolution = 0.5
        grid_size = int(x_width * 1 / resolution)
        x_grid, y_grid = np.meshgrid(np.linspace(-x_width, x_width, grid_size), np.linspace(-y_width, y_width, grid_size))
        safety_matrix = np.column_stack((x_grid.ravel(), y_grid.ravel()))

        NY = np.ones((len(points), 1))
        X_query = np.array([0, 0]).reshape((1, 2))

        k_star = self.rbf_kernel(points, safety_matrix, length_scale, sigma_f)
        K = self.rbf_kernel(points, points, length_scale, sigma_f)
        K_self = self.rbf_kernel(points, X_query, length_scale, sigma_f)
        k_inv = np.linalg.pinv(K)

        return (1 - 2 * (k_star.T @ k_inv @ NY)), grid_size, x_grid, y_grid
    
    def rbf_kernel(self,X1, X2, length_scale, sigma_f):
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * X1 @ X2.T
        return sigma_f * np.exp(-0.5 * (sqdist / length_scale**2))

    def updateCBF(self,Pgons):
        try:
        # Clear previous contour plot
            for c in self.contour.collections:
                c.remove()
            for c in self.cbf_contour.collections:
                c.remove()
        except:
            pass
        B,grid_size,x_grid,y_grid = self.CBF(Pgons)
        X_query = [0,0]
        
        # Compute H and clip
        h_grid = B.reshape(grid_size, grid_size)
        black_square = Rectangle((0, 0), 1, 0.5, color='black', zorder=10)  # position (-2, -2) and size 4x4
        self.ax.add_patch(black_square)
        
        self.contour = self.ax.contourf(x_grid, y_grid, h_grid, 20, cmap='plasma')
        self.cbf_contour = self.ax.contour(x_grid, y_grid, h_grid, [0], colors='black', linewidths=2)
        lidar_legend = Line2D([0], [0], marker='o', color='w', markerfacecolor='none', 
                              markeredgecolor='red', markersize=10, label='Lidar Points')

        # Custom black line for 0 level set
        level_set_legend = Line2D([0], [1], color='black', label='0 level set')

        # Adding custom legend to the plot
        self.ax.legend(handles=[lidar_legend, level_set_legend], loc='upper right')
 
        try:
            self.colorbar.remove()
        except:
            pass
        self.colorbar = self.fig.colorbar(self.contour)
        # Update the colorbar as well
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
    
    def updateLidar(self,points):
        if hasattr(self, 'scatter_plot'):
            self.scatter_plot.remove()
        X = points[:,0]
        Y = points[:,1]
        self.scatter_plot = self.ax.scatter(X, Y, facecolors='none', edgecolors='red', marker='o', label='Lidar Points')
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def show(self):
        plt.ioff()
        #plt.show()

    