import numpy as np
import unittest
import time

class Ackerman:
    def __init__(self, Ts, initial_x, initial_y, initial_heading):
        # robot parameters
        self.R = 0.1
        self.L = .1
        self.lf = self.L/2
        self.lr = self.L/2
        self.max_voltage = 20
        self.motor_J = 0.1
        self.motor_b = 1
        self.motor_K = 2
        self.gamma = 0
        self.vel = 0
        self.steer = 0
        self.accel = 0
        
        self.s_min = -np.pi/2
        self.s_max = np.pi/2
        self.sv_min = -np.pi/2
        self.sv_max = np.pi/2
        self.a_max = 3
        self.v_min = 0
        self.v_max = 5

        self.Ts = Ts                    # sample time of the physics

        # Robot states
        self.omega_l = 0                # left motor angular speed
        self.omega_r = 0                # right motor angular speed
        self.heading = initial_heading  # robot heading (angle)
        self.x = initial_x              # x position of the robot
        self.y = initial_y              # y position of the robot

        self.speed = 0                  # robot speed
        self.omega = 0                  # angular speed of the robot body
    
    def vehicle_dynamics_ks(self, x, u_init, lf, lr, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max):
        """
        Kinematic Bicycle Vehicle Dynamics.

                x = (x1, x2, x3, x4)
                    x1: x position in global coordinates
                    x2: y position in global coordinates
                    x3: steering angle of front wheels
                    x4: velocity in x direction
                u = (u1, u2)
                    u1: steering angle of front wheels
                    u2: acceleration

        """
        # wheelbase
        lwb = lf + lr

        # constraints clip inputs
        u = np.array([np.clip(u_init[0], sv_min, sv_max),
                            np.clip(u_init[1], v_min, v_max)])
        
        # u[0] is steer and u[1] accell
        u = u_init

        # system dynamics
        # f and g combined
        x = np.array([np.cos(x[4]),        
                      np.sin(x[4]),
                        np.tan(x[4])/lwb,0])*u[1]+np.array((0,0,0,1))*u[0]
        return x
    
    def update(self, accel, steer):
        # then update the robot's body linear speed and angular speed, then heading and x/y

        self.f = self.vehicle_dynamics_ks((self.x,self.y,self.heading + self.omega,self.speed,self.heading),(steer,accel),self.lf,self.lr,self.s_min,self.s_max,self.sv_min,self.sv_max,0,self.a_max,self.v_min,self.v_max)
        
        self.heading += self.Ts*self.f[3]
        self.x += self.Ts*self.f[0]
        self.y += self.Ts*self.f[1]

    