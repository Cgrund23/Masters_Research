from shapely.geometry import Polygon,LineString
import matplotlib.pyplot as plt
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

class RRTNorm:

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
        plt.pause(0.01)
    
    def plot_tree(self, Qs,batch_size = 10):
        """
        Efficiently visualizes the entire RRT tree while keeping all lines visible.
        """
        max_len = len(Qs)
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

            plt.pause(0.1)  # Pause to visualize updates


    def plot_final(self,path):
        color = 'black'
        if not isinstance(path, np.ndarray):
            path = np.array(path)
            color = 'green'
        self.ax.plot(path[:, 0], path[:, 1], color, alpha=0.7)
        plt.pause(1) 

    def find_path(self):
        N = 10
        start = time.time()
        Q = np.array((self.start[0],self.start[1],0)).reshape((1,3)) # parent index [x y (point num)]
        #self.init_plot()

        for k in range(0,self.max_itterations,1):
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
                #print(angle)

            # Find intersection
            does_intersect = False
            for l in range(0,self.numobjects):
                line = LineString([newPoint.flatten(),(Q[closePoint,0],Q[closePoint,1])])
                pgon = self.pgons[l]
                shapely_pgon = Polygon(pgon.get_xy())
                does_intersect = line.intersects(shapely_pgon)
                if does_intersect:
                    break
            if not does_intersect:  
                info = [newPoint[0,0],newPoint[0,1],closePoint]
                Q = np.vstack((Q,info))

                if(np.sqrt((newPoint[0,0]-self.goal[0])**2 + (newPoint[0,1]-self.goal[1])**2)<self.goal_rad):
                    break
        
        #self.plot_tree(Q,batch_size = 50)
        
                
        if (k>=self.max_itterations-1):
            OptimalPath = False
        else:
            OptimalPath = self.goal
            pointNum = int(Q[-1,2])
            while pointNum != 0:
                OptimalPath = np.vstack((OptimalPath, ((Q[pointNum, 0], Q[pointNum, 1]))))  # Append as a tuple
                pointNum = int(Q[pointNum, 2])
        end = time.time()

        #self.plot_final(OptimalPath)

        return OptimalPath,(end-start),k
    
