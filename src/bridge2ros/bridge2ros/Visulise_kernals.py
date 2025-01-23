import random
import time
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import numpy as np
from scipy.linalg import cholesky, solve, solve_triangular
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

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

    def updateCBF(self,B):
        try:
        # Clear previous contour plot
            for c in self.contour.collections:
                c.remove()
            for c in self.cbf_contour.collections:
                c.remove()
        except:
            pass
        x_width = 12
        y_width = 12
        resolution = .5    # resolution of lidar data
        grid_size = int(x_width*1/resolution)    # Grid resolution matches lidar grid
        x_grid, y_grid = np.meshgrid(np.linspace(-x_width, x_width, grid_size), np.linspace(-y_width, y_width, grid_size))
        h_grid = B.reshape(grid_size, grid_size)
        black_square = Rectangle((0, 0), 2, 1, color='black', zorder=10)  # position (-2, -2) and size 4x4
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
    
    def updateLidar(self,angle,range):
        if hasattr(self, 'scatter_plot'):
            self.scatter_plot.remove()
        X = range*np.cos(angle)
        Y = range*np.sin(angle)
        self.scatter_plot = self.ax.scatter(X, Y, facecolors='none', edgecolors='red', marker='o', label='Lidar Points')
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def show(self):
        plt.ioff()
        #plt.show()

    