# Differential drive robot simulation, intended for EE5550.
# Copyright 2024, Prof. Hamid Ossareh @ University of Vermont
# Robot dynamics, graphics, and path following with pure pursuit+AO+motor controllers.

import numpy as np
from matplotlib.patches import Polygon,CirclePolygon
from shapely.geometry import Polygon as ShapelyPolygon, Point
from ackerman import Ackerman
from robot import Robot
from controller import Controller
import draw
import pygame
from viz import MyFig
from RRT import RRT

# frame rate, and physics/control sampling periods
FPS = 60
physics_updates_per_controller_update = 5  #about 3ms
physics_updates_per_frame_update = 10      #about 1.5ms
T_phys = 1/FPS/physics_updates_per_frame_update
T_cont = 1/FPS/(physics_updates_per_frame_update/physics_updates_per_controller_update)

####################################################################################

robot = Robot(T_phys, 0, 3, 0)      # robot object
ackerman = Ackerman(T_phys,0,0,0)
controller = Controller(T_cont)     # controller object
V_r, V_l = 0, 0                     # initialize initial voltages
sensor_radius = 15                   # sensing radius for obstacle detection
#fig = MyFig(21)
goal = [-18,-18]

# Obstacles
pgons = [
    # Walls
    Polygon([[-21, -21], [21, -21], [21, -22], [-21, -22]]),
    Polygon([[-21, 21], [21, 21], [21, 22], [-21, 22]]),
    Polygon([[-21, 21], [-21, -21], [-22, -21], [-22, 21]]),
    Polygon([[21, 21], [21, -21], [22, -21], [22, 21]]),

    # Obstacles
    Polygon([[7, 2], [10, 2], [10, -10], [7, -10]]),
    Polygon([[14, 0], [13.5, 0], [13, -2], [14, -2]]),
    Polygon([[-10, -15], [-7, -15], [-7, 10], [-10, 10]]),
    Polygon([[14, 2], [15, 2], [15, -10], [14, -10]]),
    Polygon([[10, -6], [10, -10], [11.25, -10], [11, -6]]),
    Polygon([[17, 17.40],
        [11.04, 15.],
        [11.50, 13.00],
        [7.94, 13.04],
        [6.50, 12.00],
        [6.94, 8.46],
        [11.50, 6.00],
        [15.6, 6.2]])

        ]

# Helper function to compute distances from obstacles
def distance_to_polygon(pgon, x, y):
    point = Point(x, y)
    shapely_pgon = ShapelyPolygon(pgon.get_xy())
    distance = point.distance(shapely_pgon)
    nearest_point = shapely_pgon.exterior.interpolate(shapely_pgon.exterior.project(point))
    return distance, [nearest_point.x, nearest_point.y]

def sample_points_along_polygon(pgon, x, y, num_samples=10):
    """
    Returns a list of sampled points along the polygon exterior.

    pgon: Matplotlib Polygon object or array of vertices.
    x, y: Coordinates of the reference point.
    num_samples: Number of points to sample along the polygon exterior.
    """
    point = Point(x, y)
    shapely_pgon = ShapelyPolygon(pgon.get_xy())

    # Sample points along the exterior of the polygon
    exterior = shapely_pgon.exterior
    distance = point.distance(shapely_pgon)
    spaceing = np.linspace(0, exterior.length, num_samples)
    sampled_points = [exterior.interpolate(dist) for dist in spaceing]

    # Create a list of coordinates and their distances from the input point
    sampled_coords = [(pt.x, pt.y) for pt in sampled_points]
    sampled_coords = np.array(sampled_coords).reshape((-1,2))

    return distance,sampled_coords

def detect_obstacles(x, y):
    #closest_points = []
    closest_points = np.array((50,50))
    for j in range(len(pgons)):
        #dist, points = sample_points_along_polygon(pgons[j], x, y)
        dist, points = distance_to_polygon(pgons[j], x, y)
        if (dist==0):
            print('\033[31mCrash detected!\033[0m')
        elif (dist < 15):
            closest_points = np.vstack((closest_points,points))
            #closest_points.append(points)
    #return points
    return closest_points

# For logging and plotting
data = {
    'time': [0], 'x': [ackerman.x], 'y': [ackerman.y], 'target': [controller.target],
    'heading': [ackerman.heading], 'speed': [0], 'left voltage': [0], 'right voltage': [0]
}

################################ 
# new for this homework
path = np.array([[2, 2], [0, 6], [8, 9], [11,5], [13, 0], [13,-5], [12,-8], [18,-10],[18,14]])
path = path.T
# RRT_ = RRT([0,0],goal,pgons,0.5,2,10000)
# path = RRT_.find_path().T

# Find the path for plotting
tr = []
for t in np.arange(0, len(path.T), 0.01):
    x, y = np.interp(t, range(0,len(path.T)), path[0,:]), np.interp(t, range(0,len(path.T)), path[1,:])
    if not ((x,y) in tr):
        tr.append((x,y))
################################

sim_time = 0        # Elapsed time since simulation started running (after pressing spacebar)
running = False     # Spacebar has not yet been pressed
frame_skip = 1      # how many frames to skip while plotting (to make the sim run faster)
# Main loop
#fig.updateCBF(pgons)

while True: 
    # first process mouse and keyboard events. 
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
        elif event.type == pygame.KEYDOWN:
            # spacebar runs/stops the sim
            if event.key == pygame.K_SPACE:
                running = not running
            # number keys determine how fast the sim runs
            elif pygame.K_0 <= event.key <= pygame.K_9:
                frame_skip = event.key - pygame.K_0
    
    if running:
        if sim_time > 500:           # run/log up to 500 seconds
            running = not running

        # run the physics and controller at the correct sampling period
        for j in range(0, physics_updates_per_frame_update*frame_skip):
            if (j % physics_updates_per_controller_update == 0):
                # First check distance to all obstacles and store closest approach
                # If there is collision, print to terminal!
                closest_points = detect_obstacles(ackerman.x, ackerman.y)    
                V_r, V_l = controller.update(ackerman.omega_l, ackerman.omega_r, ackerman.heading, ackerman.x, ackerman.y, path, closest_points)


            ackerman.update(V_r, V_l)
            sim_time += T_phys

        new_data = {
            'time': sim_time, 'x':  ackerman.x, 'y': ackerman.y, 'target': controller.target, 
            'heading': ackerman.heading, 'speed': ackerman.speed, 'left voltage': V_l, 'right voltage': V_r
        }
        for k, v in new_data.items():
            data[k].append(v)
        
    # draw everything to the screen
    draw.draw(data, pgons, controller.target, FPS, running, events, tr)


