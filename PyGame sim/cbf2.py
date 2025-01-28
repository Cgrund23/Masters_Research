import numpy as np
from dataclasses import dataclass
import cvxpy as cp

class CBF:
    def __init__(self, lr=1.0, u_max=1.0, u_min=-1.0, gamma_max=np.pi/4, gamma_min=-np.pi/4):
        """
        Initialize the CBF with vehicle parameters.
        """
        @dataclass
        class Params:
            gamma: float = 0  # Steering angle
            theta: float = 0  # Orientation of the vehicle
            lr: float = 1    # Distance to rear axle
            u_max: float = 2
            u_min: float = 0
            gamma_max: float = np.pi/4
            gamma_min: float = -np.pi/4
        self.params = Params()

    def f(self):
        """
        Define the forced dynamics (here set to zero).
        """
        return np.zeros((4, 1))

    def g(self):
        """
        Define the control dynamics with inputs as velocity and turn angle.
        """
        g1 = np.array([
            [np.cos(self.params.gamma)],
            [np.sin(self.params.gamma)],
            [0],
            [np.sin(self.params.gamma - self.params.theta) / self.params.lr]
        ])
        
        g2 = np.array([
            [0],
            [0],
            [0],
            [1]
        ])
        
        return g1, g2

    def setPoints(self, lidar_points):
        """
        Accepts LiDAR points in [x, y] format to represent obstacles.
        """
        self.lidar_points = np.array(lidar_points)

    def cbf(self, x):
        """
        Compute the CBF value based on the distance from the closest obstacle.
        CBF h(x) > 0 means the vehicle is safe; h(x) < 0 means it is too close.
        """
        min_distance = np.min(np.sqrt((self.lidar_points[:, 0] - x[0])**2 + (self.lidar_points[:, 1] - x[1])**2))
        safe_distance = .5  # Arbitrary safe distance threshold
        h_x = safe_distance - min_distance
        return h_x

    def dcbf(self, x):
        """
        Compute the gradient of the CBF function with respect to the state.
        This is approximated as the gradient of the distance to the closest point.
        """
        closest_point = self.lidar_points[np.argmin(np.sqrt((self.lidar_points[:, 0] - x[0])**2 + (self.lidar_points[:, 1] - x[1])**2))]
        dcbf = np.zeros(4)
        dcbf[0] = (x[0] - closest_point[0])  # ∂h/∂x
        dcbf[1] = (x[1] - closest_point[1])  # ∂h/∂y
        return dcbf / np.linalg.norm(dcbf[:2])  # Normalize the gradient

    def lg_cbf(self, dcbf):
        """
        Compute Lg(h(x)) by applying dcbf to the control matrix `g`.
        """
        g1, g2 = self.g()
        g_matrix = np.hstack((g1, g2))  # Combine control dynamics into a 4x2 matrix
        lg_cbf = g_matrix.T @ dcbf.reshape(-1, 1)  # Results in a 2x1 matrix
        return lg_cbf

    def solve_qp(self, v_ref, gamma_ref, x):
        """
        Set up and solve the QP for safe control inputs based on CBF constraints.
        """
        u = cp.Variable(2)  # Two control inputs: velocity `v` and turn angle `gamma`

        # Define CBF constraint: Lg(h(x)) * u + Lf(h(x)) >= -γ * h(x)
        h_x = self.cbf(x)
        dcbf = self.dcbf(x)
        lg_cbf = self.lg_cbf(dcbf)

        # Define CBF constraint and safe margin γ
        gamma = .1
        cbf_constraint = lg_cbf.T @ u + gamma * h_x >= 0

        slack = cp.Variable()  # Slack variable
        objective = cp.Minimize(cp.norm(u - np.array([v_ref, gamma_ref]), 2) + 1000 * slack)  # Penalize slack in the cost function
        cbf_constraint = lg_cbf.T @ u + gamma * h_x + slack >= 0
        constraints = [
        cbf_constraint,
        slack >= 0,  # Slack must be non-negative
        u[0] <= self.params.u_max,
        u[0] >= self.params.u_min,
        u[1] <= self.params.gamma_max,
        u[1] >= self.params.gamma_min
]

        # Solve QP
        prob = cp.Problem(objective, constraints)
        prob.solve()

        if prob.status == cp.OPTIMAL:
            return u.value
        else:
            print("QP did not converge to a solution.")
            return np.array([0, 0])