# Differential drive robot simulation, intended for EE5550.
# Copyright 2024, Prof. Hamid Ossareh @ University of Vermont
# Robot dynamics, graphics, and path following with pure pursuit+AO+motor controllers.

import numpy as np
from matplotlib.patches import Polygon,CirclePolygon
from shapely.geometry import Polygon as ShapelyPolygon, Point, LineString
from ackerman import Ackerman
from robot import Robot
from controller import Controller
import draw
import pygame
from viz import MyFig
from RRT import RRTStar
from RRTNorm import RRTNorm
from RRTCStar import RRTCStar

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
#goal = [-23,-13]
goal = [-23, -17]

# Obstacles
pgons = [
    # Walls
    Polygon([[-25, -25], [25, -25], [25, -26], [-25, -26]]), # bottom 
    Polygon([[-25, 25], [25, 25], [25, 26], [-25, 26]]), # Top
    Polygon([[-25, 25], [-25, -25], [-26, -25], [-26, 25]]), # Left
    Polygon([[25, 25], [25, -25], [26, -25], [26, 25]]), # Right

    # Obstacles
    Polygon([[7, 2], [10, 2], [10, -10], [7, -10]]),
    Polygon([[7, 2], [10, 2], [10, 10], [7, 10]]),
    Polygon([[19,2],[19,3],[25,3],[25,2]]),
    Polygon([[-2,2],[-2,3],[17,3],[17,2]]),
    Polygon([[-2,-3],[2,-3],[2,-25],[-2,-25]]),
    Polygon([[-2,2],[-2,-3],[-3,-3],[-3,2]]),
    Polygon([[12,-12],[12,-20],[18,-20],[18,-12]]),
    Polygon([[12,12],[12,20],[18,20],[18,12]]),
    Polygon([[15,-8],[15,-6],[26,-6],[26,-8]]),
    Polygon([[0,2],[0,15],[-10,15],[-10,2]]),
    Polygon([[0,18],[0,25],[-10,25],[-10,18]]),

    Polygon([[-15,-5],[-15,-6],[-10,-6],[-10,-5]]),
    Polygon([[-18,-5],[-18,-6],[-0,-6],[-0,-5]]),

    Polygon([[-25,-10],[-25,-11],[-15,-11],[-15,-10]]),
    Polygon([[-25,-14],[-25,-15],[-15,-15],[-15,-14]]),
    Polygon([[-0,-10],[-0,-11],[-8,-11],[-8,-10]]),
    Polygon([[-0,-14],[-0,-15],[-8,-15],[-8,-14]]),

    Polygon([[-25,-20],[-25,-19],[-15,-19],[-15,-20]]),
    Polygon([[-0,-20],[-0,-19],[-8,-19],[-8,-20]]),

    Polygon([[-12,2],[-12,15],[-25,15],[-25,2]]),


    # narrow paths

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
    Returns a list of sampled points along the closest surface of the polygon.

    pgon: Matplotlib Polygon object or array of vertices.
    x, y: Coordinates of the reference point.
    num_samples: Number of points to sample along the polygon's closest surface.
    """
    point = Point(x, y)
    shapely_pgon = ShapelyPolygon(pgon.get_xy())

    # Identify the closest point on the polygon exterior
    exterior = shapely_pgon.exterior
    nearest_point = exterior.interpolate(exterior.project(point))

    # Find the closest segment of the polygon
    coords = list(exterior.coords)
    closest_segment = None
    min_distance = float('inf')
    for i in range(len(coords) - 1):  # Iterate through line segments
        segment = (Point(coords[i]), Point(coords[i + 1]))
        segment_distance = point.distance(LineString(segment))
        if segment_distance < min_distance:
            min_distance = segment_distance
            closest_segment = segment

    # Generate evenly spaced points along the closest segment
    if closest_segment is not None:
        start, end = closest_segment
        spacing = np.linspace(0, 1, num_samples)
        sampled_points = [
            (start.x + t * (end.x - start.x), start.y + t * (end.y - start.y))
            for t in spacing
        ]

        # Return the distance to the polygon and sampled points
        return min_distance, np.array(sampled_points)

    # Fallback in case something goes wrong (shouldn't happen for valid input)
    return float('inf'), np.array([])

def sample_points_along_polygonn(pgon, x, y, num_samples=10):
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

    sampled_coords = [(pt.x, pt.y) for pt in sampled_points]
    sampled_coords = np.array(sampled_coords).reshape((-1,2))
    #dist , points = distance_to_polygon(pgons[j], x, y)
    #sampled_coords = np.hstack(sampled_coords,points)
    return distance,sampled_coords

def detect_obstacles(x, y):
    #closest_points = []
    closest_points = np.array((50,50))
    for j in range(len(pgons)):
        dist, points = sample_points_along_polygon(pgons[j], x, y)
        #dist, points = distance_to_polygon(pgons[j], x, y)
        if (dist==0):
            print('\033[31mCrash detected!\033[0m')
        elif (dist < 7):
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
RRT_ = RRTStar([0,0],goal,pgons,0.01,2,10000,np.pi/2,ackerman.heading)  # RRT connect
RRT = RRTNorm([0,0],goal,pgons,0.5,2,10000,np.pi/2,ackerman.heading)   # Bland RRt
RRT__ = RRTCStar([0,0],goal,pgons,1,2,10000,np.pi/2,ackerman.heading)   # RRT star
time = []
itterations = []
fails = 0
for p in range(0,1):
    path,sec,itt = RRT_.find_path()
    time.append(sec)
    itterations.append(itt)
    try:
        if not path:
            fails =+ 1
    except:
            pass
print(min(itterations))
print(sum(itterations)/len(itterations))
print(max(itterations))

print(sum(time)/len(time))
print(max(time))
print(min(time))
print(fails)
path = np.array(path).T

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
    closest_points = detect_obstacles(ackerman.x, ackerman.y) 
    # draw everything to the screen
    draw.draw(data, pgons, controller.target, FPS, running, events, path.T, goal, closest_points)


