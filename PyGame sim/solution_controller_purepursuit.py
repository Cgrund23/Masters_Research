# Differential drive robot controller. Implements PID on two motors and pure pursuit + AO. 
# Clips desired speeds to 2 m/s, which somewhat avoids motor saturation at steady-state
# Copyright 2024, Prof. Hamid Ossareh @ University of Vermont

from PID import PID
import numpy as np


class Controller:

    def __init__(self, Ts):
        # parameters
        self.robot_R = 0.1
        self.robot_L = 0.1
        self.max_speed = 2
        self.K_phi = 2
        self.K_e = 1
        self.zero_threshold = 1e-8
        self.Ts = Ts
        self.target = np.array([0, 0])

        # motor controllers are PI with anti-windup
        self.leftWheelController = PID(Kp=1/3, Ki=10/3, Kd=0, Ts = Ts, umax = 10, umin = -10, Kaw = 1)
        self.rightWheelController = PID(Kp=1/3, Ki=10/3, Kd=0, Ts = Ts, umax = 10, umin = -10, Kaw = 1)

    # Update function. Goal: go to goal
    # measurements: left and right motor speeds, robot heading and x/y positions
    # outputs: left and right motor voltages
    def update(self, omega_l, omega_r, phi, x, y, path, points):

        # Find the point on a line segment closest to a given point.
        # P is the given point. A is the starting point of the segment.
        # B is the end point of the segment.
        def get_normal_point(P, A, B):
            # Compute the closest point on a line segment
            AP = P - A
            AB = B - A
            ABnorm = AB / np.linalg.norm(AB)
            ABcomp = np.dot(AP, ABnorm)
            
            # Project onto the line segment
            ABproj = ABnorm * ABcomp
            
            if ABcomp <= 0:
                return A
            elif ABcomp >= np.linalg.norm(AB):
                return B
            else:
                return A + ABproj

        # Find the target, a look-ahead distance ahead of the point closest to the robot on the path
        def find_target(path, P):
            record = 100000
            index = 1
            look_ahead = 2.5
            
            # Loop through path segments
            for j in range(path.shape[1] - 1):
                A = path[:, j]
                B = path[:, j + 1]
                normal_point = get_normal_point(P, A, B)
                distance = np.linalg.norm(P - normal_point)
                if distance < record:
                    record = distance
                    normal = normal_point
                    index = j

            A = path[:, index]
            B = path[:, index + 1]
            
            if np.linalg.norm(normal - B) > look_ahead:
                dir = (B - A) / np.linalg.norm(B - A)
                target = normal + dir * look_ahead
            elif np.linalg.norm(normal - path[:, -1]) < look_ahead:
                target = path[:, -1]
            else:
                tmp = look_ahead - np.linalg.norm(normal - B)
                A = path[:, index + 1]
                B = path[:, index + 2]
                dir = (B - A) / np.linalg.norm(B - A)
                target = A + dir * tmp

            return target

        def norm(x, y):
            return (x**2+y**2)**0.5

        # returns the gain adjusted on U_AO
        def Kp_obstacle(x):
            return np.clip((-x + self.max_speed), 0, self.max_speed) / max(0.0001, x)

        # assign velocity vector
        U_AO = np.zeros(2)
        for j in range(len(points)):
            p = points[j]
            position_error = np.array([p[0] - x, p[1] - y])
            velocity_vector_temp = -Kp_obstacle(np.linalg.norm(position_error)) * position_error
            U_AO += velocity_vector_temp

        # compute GTG 
        self.target = find_target(path, np.array([x, y]))            
        x_des, y_des = self.target[0], self.target[1]
        U_GTG = self.K_e*np.array([x_des - x, y_des - y])

        # clip speeds to 2m/s
        def clip_speed(u):
            norm_tmp = np.linalg.norm(u)
            if norm_tmp > self.max_speed:
                u = u / norm_tmp * self.max_speed
            return u

        U_GTG = clip_speed(U_GTG)
        U_AO  = clip_speed(U_AO)

        # Blend GTG and AO
        sigma = min(1, np.linalg.norm(U_AO)/self.max_speed)
        u = sigma * U_AO + (1 - sigma) * U_GTG
        ux, uy = u[0], u[1]
        s_des = norm(ux, uy)
        if (s_des < self.zero_threshold):
            s_des = 0
            phi_des = phi
        else:
            phi_des = np.arctan2(uy, ux)

        # proportional control for heading tracking. 
        error = phi_des - phi
        error = np.atan2(np.sin(error), np.cos(error))    # Wrap angles! 
        omega_des = error*self.K_phi

        # motor setpoints
        omega_r_des = (s_des + omega_des*self.robot_L)/self.robot_R
        omega_l_des = (s_des - omega_des*self.robot_L)/self.robot_R
        
        # update motor PIDs
        V_r = self.rightWheelController.update(omega_r_des, omega_r)
        V_l = self.leftWheelController.update(omega_l_des, omega_l)

        return V_r, V_l
    

