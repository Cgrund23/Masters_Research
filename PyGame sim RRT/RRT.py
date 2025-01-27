from shapely.geometry import Polygon,LineString
import numpy as np
import time

class RRT:

    def __init__(self,start,goal,pgons,goalrad,max_dist,max_itterations) -> None:

        self.start = start
        self.goal = goal
        self.goal_rad = goalrad
        self.max_dist = max_dist
        self.max_itterations = max_itterations
        self.pgons = pgons
        self.numobjects = len(self.pgons)
        pass

    def find_path(self):
        start = time.time()
        Q = np.array((self.start[0],self.start[1],0)).reshape((1,3)) # parent index [x y (point num) (total points) (cost to start)]
        #do stuff
        for k in range(0,self.max_itterations,1):
            # more stuff
            randPoint = np.random.uniform(-25, 25, size=(1,2))
            closest = 100000
            closePoint = 0
            # for m in range(0,Q.shape[0]):
            #     dist = np.sqrt((randPoint[0,0]-Q[m,1])**2 + (randPoint[0,1]-Q[m,2])**2)
            #     if (dist < closest):
            #         closest = dist
            #         closePoint = m
            # Compute distances from all points in Q to randPoint

            # Compute distances from all points in Q to random_point
            distances = np.sum((Q[:, 0:2] - randPoint)**2, axis=1)

            # Find the index of the closest point
            closest, closePoint = np.min(distances), np.argmin(distances)
                
            if (closest > self.max_dist):
                # pull closer
                # unit vector
                vector = randPoint - (Q[closePoint,0],Q[closePoint,1])
                newPoint = vector / np.linalg.norm(vector) * self.max_dist + (Q[closePoint,0],Q[closePoint,1])
            else:
                newPoint = randPoint

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
        if (k>=self.max_itterations):
            OptimalPath = False
        else:
            OptimalPath = self.goal
            pointNum = int(Q[-1,2])
            while pointNum != 0:
                OptimalPath = np.vstack((OptimalPath, ((Q[pointNum, 0], Q[pointNum, 1]))))  # Append as a tuple
                pointNum = int(Q[pointNum, 2])
        end = time.time()
        print(end-start)
        return OptimalPath
    
