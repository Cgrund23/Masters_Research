import numpy as np
from scipy.optimize import minimize

class CBF:
    def __init__(self, wheelbase=2.5, safe_distance=1.0, alpha=1.0, sigma=1.0):
        self.L = 1  # Wheelbase (meters)
        self.safe_distance = safe_distance  # Minimum safe distance (meters)
        self.alpha = 1  # CBF gain
        self.sigma = 2  # RBF width parameter

    def f(self, x):
        """
        Drift dynamics of the kinematic bicycle model.
        """
        v = x[3]
        theta = x[2]

        return np.array([
            v * np.cos(theta),
            v * np.sin(theta),
            0,
            0
        ])

    def g(self, x, u):
        """
        Control influence matrix of the kinematic bicycle model.
        """
        v = x[3]
        delta = u[1]

        return np.array([
            [0, 0],
            [0, 0],
            [0, v / self.L * (1 / (np.cos(delta))**2)],
            [1, 0]
        ])

    def collect_lidar_points(self, num_points=100, max_range=10.0):
        """
        Simulate LiDAR data collection. Returns a list of (x, y) points.
        """
        angles = np.linspace(-np.pi, np.pi, num_points)
        distances = max_range * np.random.random(num_points)
        lidar_points = []

        for angle, distance in zip(angles, distances):
            x = distance * np.cos(angle)
            y = distance * np.sin(angle)
            lidar_points.append((x, y))

        return np.array(lidar_points)

    def rbf(self, x, p):
        """
        Gaussian Radial Basis Function centered at point p.
        """
        distance = np.linalg.norm(x[:2] - p)
        return np.exp(-distance**2 / (2 * self.sigma**2))

    def cbf(self, x, lidar_points, weights=None):
        """
        Control Barrier Function h(x) using RBF representation of the obstacle boundary.
        """
        if weights is None:
            weights = np.ones(len(lidar_points)) / len(lidar_points)

        # Compute the RBF representation of the obstacle boundary
        phi = np.sum([w * self.rbf(x, p) for w, p in zip(weights, lidar_points)])
        h = self.safe_distance**2 - phi
        return h

    def cbf_derivative(self, x, u, lidar_points, weights=None):
        """
        Time derivative of the CBF using the affine model.
        """
        h_x = self.cbf(x, lidar_points, weights)

        # Compute the gradient of the RBF-based CBF
        dh_dx = np.zeros(4)
        if weights is None:
            weights = np.ones(len(lidar_points)) / len(lidar_points)
        for w, p in zip(weights, lidar_points):
            rbf_value = self.rbf(x, p)
            gradient = -w * rbf_value * (x[:2] - p) / (self.sigma**2)
            dh_dx[:2] += gradient

        # Affine model components
        fx = self.f(x)
        gx = self.g(x, u)

        # Compute the derivative of h(x)
        Lf_h = np.dot(dh_dx, fx)
        Lg_h = np.dot(dh_dx, gx).dot(u)

        # CBF constraint: Lf_h + Lg_h >= -alpha * h(x)
        cbf_constraint = Lf_h + Lg_h + self.alpha * h_x
        return cbf_constraint

    def qp_controller(self, x, lidar_points, u_des, weights=None):
        """
        Quadratic Program (QP) to find control inputs [a, delta] that satisfy the CBF constraint
        and minimize deviation from a desired control input.
        
        Parameters:
        - x: Current state [x, y, theta, v]
        - lidar_points: LiDAR data points
        - u_des: Desired control input [a_des, delta_des]
        - weights: Optional weights for the RBF representation
        """
        u0 = np.array([x[3], 0.0]) 

        # Define the cost function to minimize the deviation from desired control input
        def cost(u):
            return np.sum((u - u_des)**2)

        # Define CBF constraints based on RBF representation
        constraints = [{
            'type': 'ineq',
            'fun': lambda u: self.cbf_derivative(x, u, lidar_points, weights)
        }]

        # Bounds for the control inputs: [a_min, a_max], [delta_min, delta_max]
        bounds = [(-3, 3), (-np.pi/4, np.pi/4)]

        # Solve the QP problem
        result = minimize(cost, u0, constraints=constraints, bounds=bounds)

        if result.success:
            u_opt = result.x
            print(f"Optimal control input: a = {u_opt[0]:.2f}, delta = {u_opt[1]:.2f}")
            return u_opt
        else:
            print("QP optimization failed:", result.message)
            return np.array([0.0, 0.0])



