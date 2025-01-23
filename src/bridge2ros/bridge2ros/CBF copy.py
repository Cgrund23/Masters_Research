import autograd.numpy as np
from sympy import symbols
from dataclasses import dataclass
import numdifftools as nd
from autograd import jacobian
import cvxpy as cp
from qpsolvers import solve_qp


#solvers.options['show_progress'] = False




class cbf:
    # Initiate Car
    def __init__(self,param):
        @dataclass
        class params:
            pass
        self.params = param
        # Local cordinate system
        self.params.x = 0
        self.params.y = 0
        self.params.Od = {}
        self.params.Y = {}
        self.params.dsafe = 10
        pass
    
    # Dynamics
    # TODO discretise and solve 
    def f(self):
        x = self.params.v * np.cos(self.params.theta)
        y = self.params.v * np.sin(self.params.theta)
        t = self.params.theta
        return np.array((x,y,t)).reshape((3,1))
    
    def g(self):
        return np.array([0,0,1]).reshape((3,1))
    
    def x(self):
        return np.array([self.params.x, self.params.y, self.params.theta],dtype=float).reshape((3,1))
    
    def dynams(self,t,x,u):
        out = self.f() + self.g() * u
        return np.array(out).reshape(3)
        #return np.array([self.params.v * np.cos(y),self.params.v * np.sin(y),y])
    
    #TODO Changing cordinate system will alter this function
    # def setPosition(self, x):
    #     self.params.x = x[0]
    #     self.params.y = x[1]
    #     self.params.theta = x[2]
    
    #set lidar point cloud
    def setObjects(self, distance,angle):
        filtered_distance = []
        filtered_angle = []
        for k in range(len(distance)):
            if not np.isinf(distance[k]):  # If the distance is not inf, keep it
                filtered_distance.append(distance[k])
                filtered_angle.append(angle[k])

        y_lidar = filtered_distance *np.sin(filtered_angle)
        self.params.x_lidar = filtered_distance *np.cos(filtered_angle)
        
        # self.numObjects = int((len(distance)))
        # N=0
        # for k in range(self.numObjects):
        #     if distance[k] < 12:
        #         self.params.Od[N] = distance[k]
        #         self.params.Y[N] = -1
        #         N += 1
        # self.N = N
        # self.params.angle = angle

    # Functions
    #TODO Define new cbf barrier equation

    def rbf_kernel(X1, X2, length_scale, sigma_f):
        """
        Computes the RBF (Radial Basis Function) kernel between X1 and X2.
        """
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * X1 @ X2.T  # distance between points in X1 and X2
                                                                                    # note the dimentions in the sums!
                                                                                    # This is to create a matrix containing 
                                                                                    # all distances between pairs of points
        return sigma_f**2 * np.exp(-0.5 * sqdist / length_scale**2)  # Same kernel as in paper

    # function defining CBF (simple - not the one on GP Lidar paper)
    def cbf_function(self, x_test, X_train, length_scale, sigma_f, alpha,safe_dist):
        """
        Computes the CBF at certain distance.
        """
        return  self.rbf_kernel(x_test, X_train, length_scale, sigma_f) @ alpha - safe_dist
    
    def cbf(self,k):
        # scalar length never mentions value range?
        self.l = 2
        x=x
        # Direct from the paper 
        return (1 - 2*np.exp(-1/(2*self.l**2)*self.params.Od[k]))
    
    # Obtaining Lie derivatives of CBF
    # TODO look into lambda functions

    def dcbf(self,k):
        # return (-1/self.l)*self.cbf(k)
        jacobian_dcbf = jacobian(self.cbf)
        return jacobian_dcbf(self.x(),k).reshape((1,3))
    def lf_cbf(self,k):
        return self.dcbf(k) @ self.f().reshape((3,1))
    def lg_cbf(self,k):
        return self.dcbf(k) @ self.g()
        
    # Constraints/Cost
    def constraints_cost(self):

        u_ref = np.zeros((self.params.udim, 1))

        # B = self.cbf(self.x())
        # LfB = float(self.lf_cbf())
        # LgB = float(self.lg_cbf())

        # A = np.array(([float(-LgB), 0]))
        # b = np.array(([float(LfB) + float(self.params.cbfrate * B)]))
        A = np.empty((0,2), float)
        B = {}
        b = np.empty((0,1),float)
        LfB = {}
        LgB = {}
        # Make barrier for each object
        for k in range(self.N):
            B[k] = self.cbf(1,k)
            #print(B)
            print(self.lf_cbf(k)[0])
            LfB[k] = float(self.lf_cbf(k)[0])
            print(self.lg_cbf(k)[2])
            LgB[k] = float(self.lg_cbf(k)[2])
            #print(LgB[k])
            np.vstack((A, ([float(-LgB[k]), 0]))) 
            #A = np.vstack((A,[float(-LgB[k]), 0]))
            A = np.vstack((A,[float(self.params.Y[k]), 0]))
            b = np.vstack((b,[float(LfB[k]) + float(self.params.cbfrate * B[k])]))
            #print(A)
        
        # # u_min constraints 
        #A = list(A)
        #A = np.array(A)
        #A = A.items()
        A = np.vstack((A, np.hstack(([np.eye(self.params.udim), np.zeros((self.params.udim, 1))]))))
        k = self.params.u_max * np.ones((self.params.udim, 1))
        #print(k)
        b = np.vstack((b,k))
        

        # #u_max constraints
        A = np.vstack((A,np.hstack((-np.eye(self.params.udim), np.zeros((self.params.udim, 1))))))
        b = np.vstack((b,(-self.params.u_min * np.ones((self.params.udim, 1)))))

    # Cost
        
        weight_input = np.eye(self.params.udim)
        # cost = 0.5 [u' slack] H [u; slack] + f [u; slack]
        H = np.array(([float(weight_input), 0],[0, np.abs(self.params.weightslack)]))
        f_ = np.array(([float(weight_input) * float(u_ref)],[0]))
        
        #TODO This runs once with horizon 1 not what is desired
        try:
            for N in range(10):
                #print(H)
                #print(f_)
                # print(A)
                # print(b)
                x = solve_qp(H, f_, A, b, solver="clarabel")
                # Optimal control input
                u = x[0]
                self.params.weightslack = float(x[1])
                print(u)
            
            # print("x")
            # print(self.params.x)
            # print("y")
            # print(self.params.y)
            # print("Theta")
            # print(self.params.theta)
            # print("U")
            #print(x)
            #print(A)
            #self.params.weightslack = float(x[1]) 
            return u
        except Exception as e:
            print(f"An error occurred: {e}")
            return 0.0
        

        
