# Differential drive robot kinematics/dynamics
# Copyright 2024, Prof. Hamid Ossareh @ University of Vermont

import numpy as np

class Robot:
    
    def __init__(self, Ts, initial_x, initial_y, initial_heading):
        # robot parameters
        self.R = 0.1
        self.L = 0.1
        self.max_voltage = 10
        self.motor_J = 0.1
        self.motor_b = 1
        self.motor_K = 2
        self.Ts = Ts                    # sample time of the physics

        # Robot states
        self.omega_l = 0                # left motor angular speed
        self.omega_r = 0                # right motor angular speed
        self.heading = initial_heading  # robot heading (angle)
        self.x = initial_x              # x position of the robot
        self.y = initial_y              # y position of the robot

        self.speed = 0                  # robot speed
        self.omega = 0                  # angular speed of the robot body

    def update(self, V_r, V_l):
        # First saturate the motor voltages, then update the motor speeds 
        # then update the robot's body linear speed and angular speed, then heading and x/y
        V_l_sat = np.clip (V_l, -self.max_voltage, self.max_voltage)
        V_r_sat = np.clip (V_r, -self.max_voltage, self.max_voltage)
        
        self.omega_l = self.omega_l + self.Ts/self.motor_J*(self.motor_K*V_l_sat - self.motor_b*self.omega_l)
        self.omega_r = self.omega_r + self.Ts/self.motor_J*(self.motor_K*V_r_sat - self.motor_b*self.omega_r)

        self.omega = (self.omega_r - self.omega_l)*self.R/2/self.L
        self.speed = (self.omega_r + self.omega_l)*self.R/2

        self.heading = self.heading + self.Ts*self.omega
        self.x = self.x + self.Ts*self.speed*np.cos(self.heading)
        self.y = self.y + self.Ts*self.speed*np.sin(self.heading)
        
