import autograd.numpy as np
from sympy import symbols
from dataclasses import dataclass
import numdifftools as nd
from autograd import jacobian
import cvxpy as cp
from qpsolvers import solve_qp
import time
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class CBF:
    # Initialize Car
    def __init__(self, param):
        """
        Set up the car with parameters and dynamics
        """
        @dataclass
        class params:
            pass
        self.params = param
        # Local coordinate system
        self.params.x = 0
        self.params.y = 0
        self.params.Od = {}
        self.params.Y = {}
        self.params.sigma_f = 1
        self.length_scale = self.params.length_scale   # found from  loop demo

    # Update vehicle pose
    def update_pose(self, x, y, theta, v):
        self.params.x = x
        self.params.y = y
        self.params.theta = theta
        self.params.v = v
        return np.array((x, y, theta, v)).reshape((4, 1))
    
    def gamma(self,beta):
        """
        Get the steer angle from desired angle
        """
        return np.arctan2(self.params.lf+self.params.lr*np.tan(beta),self.params.lr) 
    
    def setPoints(self, points, obstacle_theta=0, obstacle_v=0):
        """
        Take closest point and make data with full state representation for each obstacle:
        x, y, theta, v.
        
        Parameters:
        - points: List of [x, y] coordinates from LiDAR for detected obstacles.
        - obstacle_theta: Assumed orientation (angle) for each obstacle in radians (default 0).
        - obstacle_v: Assumed velocity for each obstacle (default 0 if obstacles are static).
        """
        M = len(points)
        self.N = 0
        x_lidar = []
        y_lidar = []
        theta_lidar = []
        v_lidar = []
        
        points = np.array(points)
        for k in range(M):
            # Keep only points within max lidar field
            distance_to_vehicle = np.sqrt((points[k, 0] - self.params.x)**2 + (points[k, 1] - self.params.y)**2)
            if distance_to_vehicle < 15:
                self.N += 1 
                x_lidar.append(-points[k, 0])
                y_lidar.append(points[k, 1])
                
                # Assign theta and v for each obstacle point
                theta_lidar.append(obstacle_theta)
                v_lidar.append(obstacle_v)
        
        self.Y = -1 * np.ones(self.N)
        self.NY = np.ones(self.N)
        
        # Combine x, y, theta, and v into full state representation for each obstacle
        self.Poe = np.hstack((np.array(x_lidar).reshape(-1, 1),
                            np.array(y_lidar).reshape(-1, 1),
                            np.array(theta_lidar).reshape(-1, 1),
                            np.array(v_lidar).reshape(-1, 1)))


    def f(self):
        """
        The forced dynamics of the car bike model
        """
        return np.zeros([4,1])
        x = np.array([self.params.v * np.cos(self.params.theta)]).reshape((1, 1))
        y = np.array([self.params.v * np.sin(self.params.theta)]).reshape((1, 1))
        t = np.array([0, 0]).reshape((2, 1))  # theta dot, v dot
        return np.vstack((x, y, t))
    
    def g(self):
        """
        The control dynamics of the Ackerman steering car bike [a ; beta]
        """
        return np.array([
            [np.cos(self.params.gamma)],
            [np.sin(self.params.gamma)],
            [0],
            [np.sin(self.params.gamma - self.params.theta) / (self.params.lr)]
        ]), np.array([[0], [0], [0], [1]])

    def rbf_kernel(self, X1, X2, length_scale, sigma_f):
        """
        Computes the RBF (Radial Basis Function) kernel between X1 and X2.
        """
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * X1 @ X2.T
        return sigma_f * np.exp(-0.5 * (sqdist / length_scale**2))
    
    def cbf_function(self, x_test, X_train, length_scale, sigma_f, safe_dist=3.0):
        """
        Computes the CBF, ensuring a safety margin.
        h(x) > 0 when vehicle is safe; h(x) < 0 otherwise.
        """
        rbf_value = self.rbf_kernel(x_test, X_train, length_scale, sigma_f)
        h_x = safe_dist - rbf_value.sum()
        return h_x

    def compute_dcbf(self, x, X_train, length_scale, sigma_f):
        """
        Computes the gradient of the CBF function with respect to the state variables.
        """
        rbf_values = self.rbf_kernel(x, X_train, length_scale, sigma_f)
        dcbf = np.zeros_like(x)
        for i in range(X_train.shape[0]):
            diff = x - X_train[i]
            grad_rbf = -diff / (length_scale ** 2) * rbf_values[0, i]
            dcbf += grad_rbf
        return dcbf
    
    def lf_cbf_function(self, dcbf):
        """
        Derivative of the CBF function with respect to the forced dynamics.
        """
        f = self.f()
        return dcbf @ f

    def lg_cbf_function(self, dcbf):
        """
        Derivative of the CBF function with respect to the control dynamics.
        """
       
        g1, g2 = self.g()  # Assuming g returns a tuple of two 4x1 arrays, one for each input
        g_matrix = np.hstack((g1, g2))  # Stack them side-by-side to get a 4x2 matrix
    
        #    Multiply the CBF gradient (dcbf) with the control matrix `g_matrix`
        lg_cbf = g_matrix.T @ dcbf.T
        return lg_cbf

    # Constraints and Cost
    def constraints_cost(self, u_ref, x, y, theta, v):
        # Update the pose
        X_query = self.update_pose(x, y, theta, v)  # Use full state vector

        # Compute CBF constraint parameters
        K = self.rbf_kernel(self.Poe, self.Poe, self.length_scale, self.params.sigma_f)
        K_self = self.rbf_kernel(X_query.T, self.Poe, self.length_scale, self.params.sigma_f)
        k_inv = np.linalg.pinv(K)

        # Compute h (safety margin)
        h_control = 1 - 2 * (K_self @ k_inv @ -self.Y)
        dcbf = self.compute_dcbf(X_query.T, self.Poe, self.length_scale, self.params.sigma_f)

        # Define constraints A and b for QP
        A_cbf = -self.lg_cbf_function(dcbf)
        b_cbf = self.lf_cbf_function(dcbf) + h_control**3


        # Stack input constraints
        umax = np.array(self.params.u_max)
        umin = np.array(self.params.u_min)
        
        # Create QP matrices for input constraints
        A = np.vstack((A_cbf.T, np.eye(self.params.udim), -np.eye(self.params.udim)))
        b = np.vstack((b_cbf, umax.reshape(-1, 1), -umin.reshape(-1, 1)))
        
        # Define cost function
        weight_input = np.diag(np.array([1, 11]))
        H = weight_input
        f_ = -H @ u_ref  # Minimize deviation from reference
        
        # Solve QP
        try:
            u_opt = solve_qp(H, f_, A, b, solver="clarabel")
            self.u = u_opt[0]
            #self.params.gamma = self.params.gamma(float(u_opt[1]))
            return u_opt
        except Exception as e:
            print(f"An error occurred: {e}")
            return [0, 0]