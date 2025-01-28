from shapely.geometry import Polygon,LineString
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.spatial.distance import cdist
from matplotlib.patches import Polygon as MplPolygon

class RRTCStar:

    def __init__(self,start,goal,pgons,goalrad,max_dist,max_itterations,max_angle,init_heading) -> None:

        self.start = start
        self.goal = goal
        self.goal_rad = goalrad
        self.max_dist = max_dist
        self.max_itterations = max_itterations
        self.pgons = pgons
        self.numobjects = len(self.pgons)
        self.max_angle = max_angle
        self.init_heading = init_heading

        pass

    def init_plot(self):
        """
        Initializes the plot for visualization.
        """
        self.fig, self.ax = plt.subplots()  # Create figure and axes for visualization
        self.ax.clear()  # Clear the plot for fresh visualization
        self.ax.set_xlim(-30, 30)
        self.ax.set_ylim(-30, 30)
        
        # Draw obstacles
        for pgon in self.pgons:
            shapely_pgon = MplPolygon(pgon.get_xy(), closed=True, edgecolor='red', alpha=0.5)
            self.ax.add_patch(shapely_pgon)
        

        self.ax.plot(self.start[0], self.start[1], 'go', label="Start")  # Start point
        self.ax.plot(self.goal[0], self.goal[1], 'ro', label="Goal")  # Goal point
        self.ax.legend()
        plt.pause(2)
    
    def plot_tree(self, Qs, Qg,batch_size = 10):
        """
        Efficiently visualizes the entire RRT tree while keeping all lines visible.
        """
        max_len = max(len(Qs), len(Qg))
        all_lines = []  # Store all line objects for the full tree

        for i in range(0, max_len, batch_size):
            # Add a batch of lines from Qs
            for j in range(i, min(i + batch_size, len(Qs))):
                point = Qs[j]
                if int(point[2]) != 0:
                    parent = Qs[int(point[2])]
                    line, = self.ax.plot(
                        [point[0], parent[0]], [point[1], parent[1]],
                        color='blue', alpha=0.7
                    )
                    all_lines.append(line)

            # Add a batch of lines from Qg
            for j in range(i, min(i + batch_size, len(Qg))):
                point = Qg[j]
                if int(point[2]) != 0:
                    parent = Qg[int(point[2])]
                    line, = self.ax.plot(
                        [point[0], parent[0]], [point[1], parent[1]],
                        color='orange', alpha=0.7
                    )
                    all_lines.append(line)

            plt.pause(0.1)  # Pause to visualize updates    


    def plot_final(self,path):
        color = 'black'
        if not isinstance(path, np.ndarray):
            path = np.array(path)
            color = 'green'
        self.ax.plot(path[:, 0], path[:, 1], color, alpha=0.7)
        plt.pause(1) 



    def collision(self,P1,P2):
        does_intersect = False
        for p in range(0,self.numobjects):
            for l in range(0,self.numobjects):
                line = LineString([P1,P2])
                pgon = self.pgons[l]
                shapely_pgon = Polygon(pgon.get_xy())
                does_intersect = line.intersects(shapely_pgon)
                if does_intersect:
                    return True
        return False

    def smooth(self, OptimalPath):
        """
        Smooth path found by RRT iteratively to avoid points being behind others.
        """
                # Extract x and y from the optimal path
        x = OptimalPath[:, 0]
        y = OptimalPath[:, 1]

        tr = []
        # Interpolate x and y values
        for t in np.arange(0, len(OptimalPath), 0.01):
            x, y = np.interp(t, range(0,len(OptimalPath)), OptimalPath[:,0]), np.interp(t, range(0,len(OptimalPath)), OptimalPath[:,1])
            if not ((x,y) in tr):
                tr.append((x,y))

        OptimalPath = tr
        # Combine into a 2D array
        #OptimalPath = np.vstack((x, y))

        smoothed_path = [self.goal]  # Start with the first point
        i = 0  # Index for the current start point

        while i < len(OptimalPath) - 1:
            next_point_index = i + 1
            
            # Attempt to skip points for smoothing
            for j in range(i + 2, len(OptimalPath)):
                start = OptimalPath[i]
                end = OptimalPath[j]
                line = LineString([start, end])
                does_intersect = self.collision(start,end)   
                # update the farthest valid point
                if not does_intersect:
                    next_point_index = j
                else:
                    break
            
            # Append the farthest valid point to the smoothed path
            smoothed_path.append(OptimalPath[next_point_index])
            i = next_point_index 

        smoothed_path.append(self.start)

        return smoothed_path
    
    def connect(self,Qs,Qg):
        # Compute distances from all points in Qs to all points in Qg
        distances = np.array(cdist(Qs[:, 0:2], Qg[:, 0:2]))
        min_index = np.argmin(distances) 
        row, col = np.unravel_index(min_index, distances.shape)
        start = Qs[row,:2]
        end = Qg[col,:2]
        does_intersect = self.collision(start,end)
        if not does_intersect: 
            return row,col
        

    def cost(self,tree,parent, dist):
        #compute cost to start
        costs = ((tree[int(parent), 3]))
        return costs + dist
    
    def check_neighbors(self, tree, point, parent):

        """
        Finds the best parent among neighbors based on cost and collision-free connection.
        """
        new_parent_cost = tree[int(parent), 3] + np.linalg.norm(tree[int(parent), :2] - point)
        new_parent = parent

        # Compute distances and find neighbors within a radius
        distances = np.linalg.norm(tree[:, :2] - point, axis=1)
        close = np.where(distances < 6)[0]  # Radius of 3 for neighbor search

        for j in close:
            # Compute cost from neighbor
            neighbor_cost = tree[j, 3] + distances[j]
            if neighbor_cost < new_parent_cost and not self.collision(tree[j, :2], point.reshape((2,))):
                # Update parent and cost if this neighbor is better
                new_parent_cost = neighbor_cost
                new_parent = j

        return new_parent
    
    def check_neighborss(self, tree, point, parent):
        new_parent_cost = tree[int(parent),3]
        new_parent = parent
        distances = np.sum((tree[:, 0:2] - point)**2, axis=1)
        close = np.where(distances < 3)[0]
        try:
            for j in len(close):
                # check surrounding costs
                if tree[close[j],3]<new_parent_cost:
                    new_parent_cost = tree[close,3]
                    new_parent = tree[close,2]
        except:
            pass
        
        return new_parent
        


    def find_path(self):
        N = 10
        start = time.time()
        Qs = np.array((self.start[0],self.start[1],0,0)).reshape((1,4)) # parent index [x y (point num) cost]
        Qg = np.array((self.goal[0],self.goal[1],0,0)).reshape((1,4)) # parent index [x y (point num) cost]
        
        # initial trajectories
        angles = np.linspace(-self.max_angle + self.init_heading, self.max_angle + self.init_heading, N)
        points = np.array([
        (self.start[0] + self.max_dist * np.cos(angle), self.start[1] + self.max_dist * np.sin(angle)) 
        for angle in angles])
        
        # Find intersection
        for p in range(0,N):
            does_intersect = self.collision(points[p,:].flatten(),(Qs[0,0],Qs[0,1]))
            if not does_intersect:    
                info = [points[p,0],points[p,1],0,self.max_dist]
                Qs = np.vstack((Qs,info))

        angles = np.linspace(-self.max_angle - self.init_heading, self.max_angle - self.init_heading, N)
        points = np.array([
        (self.start[0] + self.max_dist * np.cos(angle), self.start[1] + self.max_dist * np.sin(angle)) 
        
        for angle in angles])
        # Find intersection
        for p in range(0,N):
            does_intersect = self.collision(points[p,:].flatten(),(Qg[0,0],Qg[0,1]))
            if not does_intersect:    
                info = [points[p,0],points[p,1],0,self.max_dist]
                Qg = np.vstack((Qg,info))

        #do stuff
        for k in range(0,self.max_itterations,1):
            if k % 2:
                Q = Qs
            else:
                Q = Qg
            # more stuff
            randPoint = np.random.uniform(-25, 25, size=(1,2))
            closest = 100000
            closePoint = 0

            # Compute distances from all points in Q to random_point
            distances = np.sum((Q[:, 0:2] - randPoint)**2, axis=1)

            # Find the index of the closest point
            closest, closePoint = np.min(distances), np.argmin(distances)
            max_dist = self.max_dist

            if (closest > max_dist):
                
                # unit vector
                vector = randPoint - (Q[closePoint,0],Q[closePoint,1])
                newPoint = vector / np.linalg.norm(vector) * self.max_dist + (Q[closePoint,0],Q[closePoint,1])
            else:
                newPoint = randPoint
            
            next_closest = int(Q[closePoint,2])

            a_start = (Q[next_closest,0],Q[next_closest,1])
            b_start = (Q[closePoint,0],Q[closePoint,1]) # b_start = a_end
            b_end = newPoint

            u = np.array([b_start[0] - a_start[0], b_start[1] - a_start[1]])  # Vector AB
            v = np.array([b_end[0,0] - b_start[0], b_end[0,1] - b_start[1]])  # Vector CD

            # Calculate dot product and magnitudes
            
            dot_product = np.dot(u, v)
            magnitude_u = np.linalg.norm(u)
            magnitude_v = np.linalg.norm(v)

            angle = np.arctan2((np.cross(u , v)),dot_product)

            if (angle > self.max_angle):
                # move to max angle 
                    # Rotate vector `v` to max_angle relative to `u`
                rotation_angle = self.max_angle - angle  # Compute the required rotation
                rotation_matrix = np.array([
                    [np.cos(rotation_angle), -np.sin(rotation_angle)],
                    [np.sin(rotation_angle), np.cos(rotation_angle)]
                ])
                v = magnitude_v * (rotation_matrix @ (v / magnitude_v))  # Rotate and rescale to original length
                newPoint = np.array([b_start[0] + v[0], b_start[1] + v[1]] ).reshape(1,2)
                #print(angle)

            if (angle < -self.max_angle):
                # move to -max angle 
                    # Rotate vector `v` to -max_angle relative to `u`
                rotation_angle = -self.max_angle - angle  # Compute the required rotation
                rotation_matrix = np.array([
                    [np.cos(rotation_angle), -np.sin(rotation_angle)],
                    [np.sin(rotation_angle), np.cos(rotation_angle)]
                ])
                v = magnitude_v * (rotation_matrix @ (v / magnitude_v))  # Rotate and rescale to original length
                newPoint = np.array([b_start[0] + v[0], b_start[1] + v[1]]).reshape(1,2)  # Update b_end position
               

            # Find intersection
            does_intersect = self.collision(newPoint.flatten(),(Q[closePoint,0],Q[closePoint,1]))
            if not does_intersect:
                cost = self.cost(Q,closePoint,min(self.max_dist,closest)) 
                parent = self.check_neighbors( Q, newPoint, closePoint)  
                info = [newPoint[0,0],newPoint[0,1],parent,cost]

                Q = np.vstack((Q,info))
                if k % 2:
                    Qs = Q
                else:
                    Qg = Q
                
                connect = self.connect(Qs,Qg)
                if connect:
                    OptimalPath = Qs[(connect[0],0)],Qs[(connect[0],1)]
                    pointNum = int(Qs[connect[0],2])
                    while pointNum != 0:
                        OptimalPath = np.vstack((OptimalPath, ((Qs[pointNum, 0], Qs[pointNum, 1]))))  # Append as a tuple
                        pointNum = int(Qs[pointNum, 2])

                    OptimalPath1 = Qg[(connect[1],0)],Qg[(connect[1],1)]
                    pointNum = int(Qg[connect[1],2])
                    while pointNum != 0:
                        OptimalPath1 = np.vstack((OptimalPath1, ((Qg[pointNum, 0], Qg[pointNum, 1]))))  # Append as a tuple
                        pointNum = int(Qg[pointNum, 2])
                    OptimalPath = np.vstack((OptimalPath1[::-1],OptimalPath))
                    break
        # self.init_plot()
        # self.plot_tree(Qs, Qg)
        # self.plot_final(OptimalPath)
                
        if (k>=self.max_itterations-1):
            OptimalPath = False

        end = time.time()

        #OptimalPath = self.smooth(OptimalPath)

        return OptimalPath,(end-start),k
    
