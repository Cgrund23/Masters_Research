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
from cvxopt import matrix, solvers
#from viz import MyFig

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
        #self.fig = MyFig(20)

        self.length_scale = self.params.length_scale   # found from  loop demo
    # Dynamics #

    def update_pose(self,x,y,theta,v):
        self.params.x = x
        self.params.y = y
        self.params.theta = np.arctan2(np.sin(theta), np.cos(theta))
        self.params.v = v
        return np.array((x,y,theta,v)).reshape((4,1))

    def f(self):
        """
        The forced dynamics of the car bike model
        """
        return np.zeros((4,1))
    
    def g(self):
        """
        The control dynamics of the ackerman steering car bike [a ; beta]
        Theta orientation of vehicle
        v speed
        beta assumes small
        gamma steer angle
        """
        return np.hstack((np.array([[np.cos(self.params.theta)],
            [np.sin(self.params.theta)],
            [np.tan(self.params.theta)/(self.params.lr)],
            [0]]),np.array([[0],[0],[0],[1]])))*self.params.dt
        
    
    def gamma(self,beta):
        """
        Get the steer angle from input angle
        """
        return np.arctan2(self.params.lf+self.params.lr*np.tan(beta),self.params.lr)    
    

    def setPoints(self,points):
        """
        Take closest point and make data
        """
        self.M = len(points)
        x_lidar = []
        y_lidar = []
        points = np.array(points)
        self.Y = -1*np.ones(self.M)
        self.NY = np.ones(self.M)
        self.Poe = points

    def rbf_kernel(self, X1, X2, length_scale, sigma_f):
        """
        Computes the RBF (Radial Basis Function) kernel between X1 and X2.
        """

        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * X1 @ X2.T  # distance between points in X1 and X2
                                                                                     # note the dimentions in the sums!
                                                                                     # all distances between pairs of points
        return sigma_f * np.exp(-0.5 * (sqdist / length_scale**2))                   # Same kernel as in paper
    
    def sqdist(self,X1,X2):
        return np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * X1 @ X2.T
    
    def CBF_derivative(self,X_query, xtrain, length_scale, sigma_f):
        points = xtrain

        NY = 1 * np.ones((len(points), 1))
        k_star = self.rbf_kernel(points, X_query, length_scale, sigma_f)
        K = self.rbf_kernel(points, points, length_scale, sigma_f)
        k_inv = np.linalg.pinv(K)
        
        # Compute the pairwise differences for x and y coordinates
        diff_x = X_query[:, 0].reshape(-1, 1) - points[:, 0].reshape(1, -1)
        diff_y = X_query[:, 1].reshape(-1, 1) - points[:, 1].reshape(1, -1)

        # Compute the derivatives of the RBF kernel with respect to x and y
        grad_k_star_x = -(diff_x / (length_scale**2)) * k_star.T
        grad_k_star_y = -(diff_y / (length_scale**2)) * k_star.T

        # Compute the derivative of the CBF function
        grad_h_x = -2 * (grad_k_star_x @ k_inv @ NY).flatten()
        grad_h_y = -2 * (grad_k_star_y @ k_inv @ NY).flatten()

        gradient_magnitude = np.sqrt(grad_h_x**2 + grad_h_y**2)

        return gradient_magnitude

     
    def lg_cbf_function(self,dcbf):
        """
        Derivitive of the cbf function by the Input dynamics
        """
        g = self.g()[:2, :]
        return dcbf * g



    def constraints_cost(self, u_ref, x, y, theta, v, N):
        """
        Compute the optimal control input using a prediction horizon and CVXOPT.
        """
        # Initialize problem parameters
        state_dim = 4  # Assuming [x, y, theta, v]
        control_dim = 2  # Assuming [a, beta]
        X_query = self.update_pose(x,y,self.params.gamma,theta).T[:, :2]
        x_k = np.array([x, y, theta, v]).reshape((state_dim, 1))  # Current state
        

        # Weight matrices for cost function

        R = np.diag([10000, 1])        # Control input weights

        # Define matrices for horizon
        H = np.zeros((N * control_dim, N * control_dim))  # Quadratic cost matrix
        f = np.zeros((N * control_dim, 1))               # Linear cost vector
        G = []
        h = []

        # Dynamics and constraints for the horizon
        x_pred = x_k.copy()  # Initialize predicted state
        for k in range(N):
            # Compute cost for current time step
            u_k_ref = u_ref  # Assuming reference input is constant
            H_k = R
            f_k = -R @ u_k_ref

            # Add to the horizon-wide cost matrix/vector
            H[k * control_dim:(k + 1) * control_dim, k * control_dim:(k + 1) * control_dim] = H_k
            f[k * control_dim:(k + 1) * control_dim, 0] = f_k.flatten()

            # State dynamics: x_{k+1} = f(x_k) + g(x_k) * u_k
            g_k = self.g()
            x_next =   x_k + g_k

            # CBF constraint for safety
            X_query = x_next[:2].T
            dcbf = self.CBF_derivative(X_query, self.Poe, self.length_scale, 1)
            
            
            K = self.rbf_kernel(self.Poe,self.Poe,self.length_scale,self.params.sigma_f)
            K_self = self.rbf_kernel(self.Poe,X_query,self.length_scale,self.params.sigma_f)
            self.k_inv = np.linalg.pinv(K)
        

            # Compute H and clip
            h_control = 1-2*(K_self.T @ self.k_inv @ self.NY) 
            h_control = np.clip(h_control, -1, 1)
            A_cbf = -self.lg_cbf_function(dcbf)  - (h_control**3) # Derivative of the CBF wrt control
            b_cbf = np.zeros((2, 1))

             # Pad A_cbf to match horizon-wide decision variable structure
            G_k = np.zeros((2, N * control_dim))
            G_k[:, k * control_dim:(k + 1) * control_dim] = A_cbf
            G.append(G_k)
            h.append(b_cbf)

            # Predict next state
            x_k = x_next

        # Convert cost and constraints to cvxopt format
        H_cvx = matrix(H, tc='d')
        f_cvx = matrix(f, tc='d')
        G_cvx = matrix(np.vstack(G), tc='d')  # Stack all G matrices
        h_cvx = matrix(np.vstack(h), tc='d')  # Stack all h vectors

        # Solve the QP problem
        try:
            sol = solvers.qp(H_cvx, f_cvx, G_cvx, h_cvx)
            U = np.array(sol['x']).flatten()  # Extract the solution
            self.u = U[:control_dim]  # Return only the first control input
            return U[:control_dim]  # Apply only the first control action
        except Exception as e:
            print(f"An error occurred: {e}")
            return [0, 0]

    # Constraints/Cost
    def constraints_costt(self,u_ref,x,y,theta,v):

        # Create variables for optimisation 
        A = np.empty((0,2), float)
        B = {}
        b = np.empty((0,1),float)
        LfB = {}
        LgB = {}

        # referance 
        u_ref = np.array(u_ref)
        X_query = self.update_pose(x,y,self.params.gamma,theta).T[:, :2]
        SF = 175

        #self.length_scale = np.sqrt(np.linalg.norm(self.sqdist(self.Poe,X_query),ord=1)/(np.log(self.M*SF)))/4
        #self.length_scale = np.sqrt(np.linalg.norm(self.sqdist(self.Poe,X_query),ord=2)/(2*SF))
        #print(self.length_scale)
        
        #self.length_scale=0.7
        # K  matrixies
        K = self.rbf_kernel(self.Poe,self.Poe,self.length_scale,self.params.sigma_f)
        K_self = self.rbf_kernel(self.Poe,X_query,self.length_scale,self.params.sigma_f)
        self.k_inv = np.linalg.pinv(K)
        

        # Compute H and clip
        h_control = 1-2*(K_self.T @ self.k_inv @ self.NY) 
        h_control = np.clip(h_control, -1, 1)

        dcbf = self.CBF_derivative(X_query,self.Poe,self.length_scale,1)

        A = -self.lg_cbf_function(dcbf) - (h_control**3)
        A = A.reshape(2, 2)  # Ensure A is (2, 2)
        b = np.zeros((2, 1))
         
        # umax constraints
        k = np.eye(self.params.udim)
        A = np.vstack((A,k))
        k = np.array((self.params.u_max))
        b = np.vstack((b.reshape((b.size,1)),k.reshape((k.size,1))))

        # u_min constraints
        A = np.vstack((A,-np.eye(self.params.udim)))
        k = np.array((self.params.u_min))
        b = np.vstack((b,-k.reshape((k.size,1))))
        
        weight_input = np.diag(np.array((100,1)))
        H = weight_input
        f_ = ((weight_input) @ -(u_ref) )
      
        #  Optimal control input
        try:  
            x = solve_qp(H.astype('float'), f_, A, b, solver = "clarabel") 
            self.u = x[0]
            #self.params.gamma = self.gamma(float(x[1]))
            return x
        except Exception as e:
            print(f"An error occurred: {e}")
            return [0,0]

     

    def expanded_horizon():
        pass        
