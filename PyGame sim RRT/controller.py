# Differential drive robot controller. Implements PID on two motors and pure pursuit + AO. 
# Clips desired speeds to 2 m/s, which somewhat avoids motor saturation at steady-state
# Copyright 2024, Prof. Hamid Ossareh @ University of Vermont

from PID import PID
import numpy as np
import scipy.interpolate as sci
from CBF import CBF
#from viz import MyFig
from oldCBF import cbf


class Controller:

    def __init__(self, Ts):
        # parameters
        self.robot_R = 0.1
        self.robot_L = 0.1
        self.max_speed = 2
        self.K_phi = 2
        self.K_e = 1
        self.zero_threshold = 0.3
        self.Ts = Ts
        self.target = np.array([0, 0])
        self.index = 0
        self.end = False

        class params():
            # for qp
            dt: float = Ts # 100ms

            # cbf params
            Y: set = {}
            sigma_f: float = 1
            length_scale: float = .1 # found from  loop demo will tune then adaptive

            # Car info

            xdim: float = 4
            udim: float = 2
            lf: float = .1
            lr: float = .1
 
            u_max: float = np.array([2, 1.57/.5]) # max accel, angle
            u_min: float = np.array([0.1, -1.54/.5]) # min accel, angle

            # Initial state
            
            x0: float = 0   # Start x
            y0: float = 0   # Start y
            theta0: float = 0 # start theta
            v0: float = 0.0 # initial velocity
            state: float = [x0,y0,theta0,v0] # starting state vector

            beta: float = 0
            gamma: float = 0 
            
            # waypoints
            wx: float = [ 20, 30, 50.0]
            wy: float = [ 20, 20, 10.0]          

        self.params = params # store structure
        self.CBFobj = CBF(self.params) # pass structure to car
        self.oldobj = cbf(self.params)

        # motor controllers are PI with anti-windup
        self.leftWheelController = PID(Kp=1/3, Ki=10/3, Kd=0, Ts = Ts, umax = 20, umin = -20, Kaw = 1)
        self.rightWheelController = PID(Kp=1/3, Ki=10/3, Kd=0, Ts = Ts, umax = 20, umin = -20, Kaw = 1)

    # Update function. Goal: go to goal
    # measurements: left and right motor speeds, robot heading and x/y positions
    # outputs: left and right motor voltages
    def update(self, omega_l, omega_r, phi, x, y, path, points):
        
        # Find the point on a line segment closest to a given point.
        # P is the given point. A is the starting point of the segment.
        # B is the end point of the segment.
        def get_normal_point(P, A, B):
            a = P - A
            b = B - A
            # ensure on segment
            length = np.dot(a, b) / np.dot(b, b)
            length = max(0, min(1, length)) # ensure Q doesnt leave the path
            return A + length * b
                   
# Find the target, a look-ahead distance ahead of the point closest to the robot on the path
        def find_target(path, P):
            if self.end:
                self.target = path[:,-1]
                return self.target
            
            norm_path = np.zeros((2,path.shape[1]-1))

            for i in range(path.shape[1]-1):
                norm_path[:,i] = get_normal_point(P,path[:,i],path[:,i+1])
                 
            look_ahead = 2.5

            self.Q = np.array([1000.0,1000.0])

            for i in range(norm_path.shape[1]):
                # Find the closest Q smallest dist
                if (np.linalg.norm(P-norm_path[:,i]) < (np.linalg.norm(P-self.Q))):
                    self.Q = norm_path[:,i]
                    self.index = i
            try:
                segment_1 = path[:,self.index+1] - path[:,self.index]
                segment_2 = path[:,(self.index+2)] - path[:,self.index+1]

                if (np.linalg.norm(self.Q-path[:,self.index+1]) < look_ahead):
                    # means target is on next section
                    # dist takes the differance such that the target point is lookahead distance on path
                    dist = look_ahead - np.linalg.norm(P-path[:,self.index+1])
                    # Find the temp target
                    temp_target = segment_2/np.linalg.norm(segment_2) * dist + path[:,self.index+1]
                    # Take unit vector and amplify by lookahead so that magnatude is lookahead
                    self.target = (temp_target-self.Q)/np.linalg.norm(temp_target-self.Q) * look_ahead + self.Q
                else:
                    # get target along segment
                    self.target = segment_1/np.linalg.norm(segment_1) * look_ahead + self.Q
                    #self.target =  (path[:,self.index+1] - self.Q) / np.linalg.norm((path[:,self.index+1] - self.Q)) * look_ahead + self.Q
            except:
                # will fail when trying to look ahead to segment after end goal thus
                self.target = segment_1/np.linalg.norm(segment_1) * look_ahead + self.Q   
                if (np.linalg.norm(abs(self.target - path[:,-1]))) < 0.01:
                    self.target = path[:,-1]
                    self.end = True

            return self.target



# # Find the target, a look-ahead distance ahead of the point closest to the robot on the path
#         def find_target(path, P):
#             if self.end:
#                 self.target = path[:,-1]
#                 return self.target
            
#             norm_path = np.zeros((2,path.shape[1]-1))

#             for i in range(path.shape[1]-1):
#                 norm_path[:,i] = get_normal_point(P,path[:,i],path[:,i+1])
                 
#             look_ahead = 2.5

#             self.Q = np.array([1000.0,1000.0])

#             for i in range(norm_path.shape[1]):
#                 # Find the closest Q smallest dist
#                 if (np.linalg.norm(P-norm_path[:,i]) < (np.linalg.norm(P-self.Q))):
#                     self.Q = norm_path[:,i]
#                     self.index = i
#             try:
#                 segment_1 = path[:,self.index+1] - path[:,self.index]
#                 segment_2 = path[:,(self.index+2)] - path[:,self.index+1]

#                 if (np.linalg.norm(self.Q-path[:,self.index+1]) < look_ahead):
#                     dist = look_ahead - np.linalg.norm(P-path[:,self.index+1])
           
#                     temp_target = segment_2/np.linalg.norm(segment_2) * dist + path[:,self.index+1]

#                     self.target = (temp_target-self.Q)/np.linalg.norm(temp_target-self.Q) * look_ahead + self.Q
#                 else:
#                     # get target along segment
#                     self.target = segment_1/np.linalg.norm(segment_1) * look_ahead + self.Q
#                     #self.target =  (path[:,self.index+1] - self.Q) / np.linalg.norm((path[:,self.index+1] - self.Q)) * look_ahead + self.Q
#             except:
#                 # will fail when trying to look ahead to segment after end goal thus
#                 self.target = segment_1/np.linalg.norm(segment_1) * look_ahead + self.Q   
#                 if (np.linalg.norm(abs(self.target - path[:,-1]))) < 0.01:
#                     self.target = path[:,-1]
#                     self.end = True

#             return self.target



        def norm(x, y):
            return (x**2+y**2)**0.5

        # returns the gain adjusted on U_AO
        def Kp_obstacle(x):
            return np.clip((-x + self.max_speed), 0, self.max_speed) / max(0.0001, x)

        
        # compute GTG 
        P = x,y
        x_des, y_des = find_target(path, P)
        # x_des = 12
        # y_des = 0
        U_GTG = self.K_e*np.array([x_des - x, y_des - y])

        # clip speeds to 2m/s
        def clip_speed(u):
            norm_tmp = np.linalg.norm(u)
            if norm_tmp > self.max_speed:
                u = u / norm_tmp * self.max_speed
            return u

        U_GTG = clip_speed(U_GTG)
        v_des = np.linalg.norm(U_GTG)
        phi_des = np.arctan2(U_GTG[1], U_GTG[0])

        #u = self.oldobj.constraints_cost(points,[x,y,v_des],phi_des)
        error = phi_des - phi
        error = np.arctan2(np.sin(error), np.cos(error))    # Wrap angles! 

        self.CBFobj.setPoints(points)
        u = self.CBFobj.constraints_cost([v_des , error],x,y,phi,np.linalg.norm(U_GTG),5)
        print('location')
        print(x,y)

        if np.abs(np.linalg.norm(U_GTG/self.K_e)) < 1:
            u = [0,0]

        s_des = u[0]
        if (s_des < .000):
            s_des = 0
            phi_des = 0
        else:
            phi_des = u[1]

        
        
        # proportional control for heading tracking. 
        error = phi_des - phi
        error = np.arctan2(np.sin(error), np.cos(error))    # Wrap angles! 
        omega_des = error*self.K_phi

        # motor setpoints
        omega_r_des = (s_des + omega_des*self.robot_L)/self.robot_R
        omega_l_des = (s_des - omega_des*self.robot_L)/self.robot_R
        
        # update motor PIDs
        V_r = self.rightWheelController.update(omega_r_des, omega_r)
        V_l = self.leftWheelController.update(omega_l_des, omega_l)

        return s_des, phi_des
    

