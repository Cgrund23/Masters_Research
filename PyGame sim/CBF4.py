import autograd.numpy as np
from dataclasses import dataclass
from scipy.linalg import pinv
import cvxpy as cp
from qpsolvers import solve_qp

class CBF:
    def __init__(self, param):

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
        self.params = param
        self.length_scale = param.length_scale

    def setPoints(self,points):
        """
        Take closest point and make data
        """
        M = len(points)
        self.N = 0
        x_lidar = []
        y_lidar = []
        points = np.array(points)
        for k in range(M):
        # Keep only points within max lidar field
            if (np.sqrt((points[k, 0] - self.params.x)**2 + (points[k , 1] - self.params.y)**2) < 15):  
                                           
                self.N += 1 
                x_lidar.append(points[k,0])
                y_lidar.append(points[k,1])

        self.Y = -1*np.ones(self.N)
        self.NY = np.ones(self.N)
        self.Poe = np.hstack((x_lidar,y_lidar)).reshape((-1,1))

    def update_pose(self, x, y, theta, v):
        self.params.x = x
        self.params.y = y
        self.params.theta = theta
        self.params.v = v
        return np.array([x, y, theta, v]).reshape((4, 1))

    def f(self):
        x_dot = self.params.v * np.cos(self.params.theta)
        y_dot = self.params.v * np.sin(self.params.theta)
        theta_dot = 0
        v_dot = 0
        return np.array([x_dot, y_dot, theta_dot, v_dot]).reshape((4, 1))

    def g(self):
        lr = self.params.lr
        return np.array([
            [0, -np.sin(self.params.theta)],
            [0, np.cos(self.params.theta)],
            [0, self.params.v / lr],
            [1, 0]
        ])

    def rbf_kernel(self, X1, X2, length_scale, sigma_f):
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * X1 @ X2.T
        return sigma_f * np.exp(-0.5 * (sqdist / length_scale**2))

    def cbf_function(self, x_test, X_train, length_scale, sigma_f, safe_dist=3.0):
        rbf_value = self.rbf_kernel(x_test, X_train, length_scale, sigma_f)
        h_x = safe_dist - rbf_value.sum()
        return h_x

    def compute_dcbf(self, x, X_train, length_scale, sigma_f):
        rbf_values = self.rbf_kernel(x, X_train, length_scale, sigma_f)
        dcbf = np.zeros_like(x)
        for i in range(X_train.shape[0]):
            diff = x - X_train[i]
            grad_rbf = -diff / (length_scale ** 2) * rbf_values[0, i]
            dcbf += grad_rbf
        return dcbf

    def compute_lie_derivatives(self, dcbf):
        L_f_h = dcbf.T @ self.f()
        L_g_h = dcbf.T @ self.g()
        return L_f_h, L_g_h

    def constraints_cost(self, u_ref, x, y, theta, v):
        u_ref = np.array(u_ref)
        X_query = self.update_pose(x, y, theta, v).T

        dcbf = self.compute_dcbf(X_query.T, self.Poe, self.length_scale, self.params.sigma_f)
        L_f_h, L_g_h = self.compute_lie_derivatives(dcbf)

        # CBF constraint: L_f_h + L_g_h @ u + h >= 0
        A = -L_g_h.reshape(1, -1)
        b = -L_f_h - 0.5 * (self.cbf_function(X_query.T, self.Poe, self.length_scale, self.params.sigma_f))

        # Input constraints
        A = np.vstack((A, np.eye(self.params.udim), -np.eye(self.params.udim)))
        b = np.vstack((b, (self.params.u_max).reshape((-1,1)), (-self.params.u_min).reshape((-1,1))))

        # Quadratic cost: min (u - u_ref)^T Q (u - u_ref)
        Q = np.diag([1, 10])
        H = 2 * Q
        f = -2 * Q @ u_ref

        try:
            u_opt = solve_qp(H, f, A, b,solver = "clarabel")
            #self.params.gamma = self.gamma(float(u_opt[1]))
            return u_opt
        except Exception as e:
            print(f"QP Solver Error: {e}")
            return np.zeros(self.params.udim)
        