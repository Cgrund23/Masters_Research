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
    # Initiate Car
    def __init__(self,param):
        """
        Set up the car with parameters and dynamics
        """
        @dataclass
        class params:
            pass
        self.params = param
        # Local cordinate system
        self.params.x = 0
        self.params.y = 0
        self.params.Od = {}
        self.params.Y = {}
        self.params.sigma_f = 1
        
        pass
        self.length_scale = self.params.length_scale   # found from  loop demo
    # Dynamics #
    def update_pose(self,x,y,theta,v):
        self.params.x = x
        self.params.y = y
        self.params.theta = theta
        self.params.v = v
        return np.array((x,y,theta,v)).reshape((4,1))

    def f(self):
        """
        The forced dynamics of the car bike model
        """
        x = np.array([self.params.v * np.cos(self.params.theta)]).reshape((1,1))
        y = np.array([self.params.v * np.sin(self.params.theta)]).reshape((1,1))
        t = np.array([0, 0]).reshape((2,1)) # theta dot v dot
        return np.vstack((x,y,t))
    
    def g(self):
        """
        The natrual dynamics of the ackerman steering car bike [a ; beta]
        Theta orientation of vehicle
        v speed
        beta assumes small
        gamma steer angle
        """
        return np.array([
            [0, -np.sin(self.params.theta)],
            [0, np.cos(self.params.theta)],
            [0, self.params.v/(self.params.lr)],
            [1 , 0] 
        ])
        
    
    def gamma(self,beta):
        """
        Get the steer angle from desired angle
        """
        return np.arctan2(self.params.lf+self.params.lr*np.tan(beta),self.params.lr)    
    
    def x(self):
        """
        Returns the location of the car and angle of tires / Fornow x,y 0 always
        """
        return np.array([self.params.x, self.params.y, self.params.theta],dtype=float).reshape((3,1))
    
    def setObjects(self,distance,angle):                               
        """
        Take all lidar points and turn them into data
        """
        M = len(distance)   # Total Number of possible lidar data points
        self.N = 0          # Total number of points in range

        # Instantiate matrix
        filtered_distance = []
        filtered_angle = []

        for k in range(M):
        # Keep only points within max lidar field
            if not np.isinf(distance[k]):                               
                self.N += 1
                filtered_distance.append(distance[k])
                filtered_angle.append(angle[k]) 
        
        # Create value and distance to plant of all points
        self.Y = -1*np.ones(self.N)
        self.NY = np.ones(self.N)
        self.Dist = np.array(filtered_distance).reshape((self.N,1))
        
        # Convert to x y cordinates
        # Transform LiDAR points to vehicle coordinate frame
        x_lidar = np.array(np.array(filtered_distance) * np.cos(np.array(filtered_angle) + self.params.theta)+self.params.x).reshape((self.N, 1))
        y_lidar = np.array(np.array(filtered_distance) * np.sin(np.array(filtered_angle) + self.params.theta)+self.params.y).reshape((self.N, 1))

        #fill = np.zeros((self.N,1))
        self.Poe = np.hstack((x_lidar,y_lidar))#,np.array(filtered_angle).reshape((self.N,1)),fill))


    def rbf_kernel(self, X1, X2, length_scale, sigma_f):
        """
        Computes the RBF (Radial Basis Function) kernel between X1 and X2.
        """
        s1 = np.sum(X1**2, 1)
        s2 = np.sum(X2**2, 1) 
        s3 = 2 * X1 @ X2.T
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * X1 @ X2.T  # distance between points in X1 and X2
                                                                                     # note the dimentions in the sums!
                                                                                     # all distances between pairs of points
        return sigma_f * np.exp(-0.5 * (sqdist / length_scale**2))                   # Same kernel as in paper

    def cbf_functionn(self, x_test, X_train, length_scale, sigma_f, safe_dist=3.0):
        """
        Computes the CBF, ensuring a safety margin.
        h(x) > 0 when vehicle is safe; h(x) < 0 otherwise.
        """
        rbf_value = self.rbf_kernel(x_test, X_train, length_scale, sigma_f)
        h_x = safe_dist - rbf_value.sum()  # Subtracts the influence of obstacles on safety distance
        return h_x
    
    def cbf_function(self, x_test, X_train, length_scale, sigma_f):
        """
        Computes the CBF
        """
        # Compute kernel matrices
        K = self.rbf_kernel(self.Poe, self.Poe, self.length_scale, self.params.sigma_f)
        K_star = self.rbf_kernel(self.Poe, x_test, self.length_scale, self.params.sigma_f)

        # Predictive mean of the Gaussian Process
        L = np.linalg.cholesky(K + 1e-6 * np.eye(K.shape[0]))  # Add a small regularization term
        K_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(K.shape[0])))

        #K_inv = np.linalg.pinv(K)
        mean_prediction = K_star.T @ K_inv @ (-self.Y)

        # Compute the CBF value
        h_x = 1 - 2 * mean_prediction
        return h_x
        #return  self.rbf_kernel(x_test, X_train, length_scale, sigma_f) @ alpha - safe_dist

    def compute_dcbf(self, x_test, X_train, length_scale, sigma_f):
        """
        Computes the gradient of the CBF function with respect to the state variables using adaptive length_scale.
        """


        K = self.rbf_kernel(self.Poe, self.Poe, self.length_scale, self.params.sigma_f)
        K_star = self.rbf_kernel(self.Poe, x_test, self.length_scale, self.params.sigma_f)
        L = np.linalg.cholesky(K + 1e-6 * np.eye(K.shape[0]))  # Add a small regularization term
        K_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(K.shape[0])))

        #K_inv = np.linalg.pinv(K)

        # Gradient of the GP predictive mean
        dcbf = np.zeros_like(x_test)
        for i in range(self.Poe.shape[0]):
            diff = x_test - self.Poe[i]
            grad_rbf = -diff / (self.length_scale ** 2) * K_star[i, 0]
            dcbf += grad_rbf * (-self.Y[i]) * K_inv[i, i]

        return dcbf

    
    def lf_cbf_function(self,dcbf):
        """
        Derivitive of the cbf function by the forced dynamics
        """
        f = self.f()
        f = 0
        return  dcbf * f
     
    def lg_cbf_function(self,dcbf):
        """
        Derivitive of the cbf function by the Input dynamics
        """
        g = self.g()
        return g * dcbf 

    # Constraints/Cost
    def constraints_cost(self,u_ref,x,y,theta,v):
        # Create grid of safe and unsafe
        # x_width = 12
        # y_width = 12
        # resolution = .5    # resolution of lidar data
        # grid_size = int(x_width*1/resolution)    # Grid resolution matches lidar grid
        # x_grid, y_grid = np.meshgrid(np.linspace(-x_width, x_width, grid_size), np.linspace(-y_width, y_width, grid_size))
        # safety_matrix = np.column_stack((x_grid.ravel(), y_grid.ravel()))
        
        # Create variables for optimisation 
        A = np.empty((0,2), float)
        B = {}
        b = np.empty((0,1),float)
        LfB = {}
        LgB = {}

        u_ref = np.array(u_ref)
        X_query = self.update_pose(x,y,theta,v).T
        X_query = X_query[:,:2]

        # K  matrixies
        K = self.rbf_kernel(self.Poe,self.Poe,self.length_scale,self.params.sigma_f)
        K_self = self.rbf_kernel(self.Poe,X_query,self.length_scale,self.params.sigma_f)
        #k_inv = np.linalg.pinv(K)
        
        # Compute H and clip
        #h_control = 1-2*(K_self.T @ k_inv @ -self.Y)
        h_control = self.cbf_function(X_query,self.Poe,self.params.length_scale,1)
        h_control = np.clip(h_control, -1, 1)
        
        dkdp = -1/self.length_scale*K_self 
        #dcbf = self.Y.T @ k_inv @ dkdp

        dcbf = self.compute_dcbf(X_query,self.Poe,self.length_scale,1)
        
        # b = self.lg_cbf_function(dcbf) 
        # b = np.array(b @ u_ref)
         
        # A = -self.lf_cbf_function(dcbf) 
        # A -= h_control**3
        A = -self.lg_cbf_function(dcbf).reshape((4,2)) @ np.diag(u_ref) - h_control**3
        b = self.lf_cbf_function(dcbf).T + self.update_pose(x,y,theta,v).T * h_control**3
        b = np.zeros((4,1))
        

         
        # umax constraints
        k = np.eye(self.params.udim)
        A = np.vstack((A,k))
        k = np.array((self.params.u_max))
        b = np.vstack((b.reshape((b.size,1)),k.reshape((k.size,1))))

        # u_min constraints
        A = np.vstack((A,-np.eye(self.params.udim)))
        k = np.array((self.params.u_min))
        b = np.vstack((b,-k.reshape((k.size,1))))
        
        weight_input = np.diag(np.array((1,11)))
        H = weight_input
        f_ = -(weight_input) @ (u_ref) 
        #f_ = np.array([1,0,1,1])
      
        #  Optimal control input
        try:  
            x = solve_qp(H, f_, A, b, solver = "clarabel") 
            print(x[1]-u_ref[1])  
            self.u = x[0]
            #TODO update from imu data
            #self.updateState(x[1],x[0])
            self.params.gamma = float(x[1])

            # x = [1,1]
            # k_star = self.rbf_kernel(self.Poe,safety_matrix,self.length_scale,self.params.sigma_f)
            # h_world =  1-2*(k_star.T @ k_inv @ - self.Y)
            h_world = 5
            return x,[h_control,dcbf]
        except Exception as e:
            print(f"An error occurred: {e}")
            return u_ref,h_world,dcbf
        

        
