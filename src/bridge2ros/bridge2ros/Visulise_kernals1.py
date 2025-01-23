import random
import time
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import numpy as np
from scipy.interpolate import griddata
from matplotlib.patches import Rectangle

class MyFig:
    def __init__(self, grid_size=12) -> None:
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.grid_size = grid_size
        self.x_width = grid_size
        self.y_width = grid_size
        resolution = .5    # resolution of lidar data
        grid_size = int(self.x_width*1/resolution)    # Grid resolution matches lidar grid
        self.x_grid, self.y_grid = np.meshgrid(np.linspace(-self.x_width, self.x_width, grid_size), np.linspace(-self.y_width, self.y_width, grid_size))

        self.Z = np.sin(self.x_grid) * np.cos(self.y_grid)
        self.ax.set_title('CBF vs Barrier')

        self.contour = self.ax.contourf(self.x_grid, self.y_grid, self.Z, cmap='plasma', levels=20)
        self.colorbar = self.fig.colorbar(self.contour)

        # For lidar points
        self.scatter_plot = None

        # Create a smooth transition buffer
        self.prev_h_grid = np.zeros((24,24))
        self.transition_frames = 3  # Number of frames to transition between contours

    def smooth_transition(self, old_grid, new_grid, steps):
        """Smoothly transition between two grids over `steps` frames."""
        for i in range(steps):
            intermediate_grid = old_grid + (new_grid - old_grid) * (i / steps)
            self.update_contour(intermediate_grid)
            

    def update_contour(self, h_grid):
        """Update the contour plot."""
        # Remove previous contours
        for c in self.contour.collections:
            c.remove()

        self.contour = self.ax.contourf(self.x_grid, self.y_grid, h_grid, levels=20, cmap='plasma')
        self.fig.canvas.draw_idle()

    def updateCBF(self, B):
        """Smoothly update the contour with new CBF data."""
        
        x_width = 12
        y_width = 12
        resolution = .5
        grid_size = int(x_width * 1 / resolution)

        x_grid, y_grid = np.meshgrid(
            np.linspace(-x_width, x_width, grid_size),
            np.linspace(-y_width, y_width, grid_size)
        )

        h_grid = B.reshape(grid_size, grid_size)

        # Add obstacle
        black_square = Rectangle((0, 0), 2, 1, color='black', zorder=10)
        self.ax.add_patch(black_square)

        # Smooth transition from previous contour to the new one
        #self.smooth_transition(self.prev_h_grid, h_grid, self.transition_frames)
        self.update_contour(h_grid)
        # Store the new grid for the next update
        self.prev_h_grid = h_grid

        # Remove and update the colorbar
        try:
            self.colorbar.remove()
        except:
            pass
        self.colorbar = self.fig.colorbar(self.contour)

    def updateLidar(self, angle, range):
        """Smoothly update the lidar points."""
        if self.scatter_plot:
            self.scatter_plot.remove()

        # Calculate lidar points
        X = range * np.cos(angle)
        Y = range * np.sin(angle)

        self.scatter_plot = self.ax.scatter(X, Y, facecolors='none', edgecolors='red', marker='o', label='Lidar Points')
        self.fig.canvas.draw_idle()
        

    def show(self):
        """Show the plot continuously without timing out."""
        plt.ioff()
        plt.show()