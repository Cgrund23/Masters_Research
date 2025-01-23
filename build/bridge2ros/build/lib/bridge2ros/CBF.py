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
        self.params.x = self.params.x0
        self.params.y = self.params.y0
        pass
    
    # Dynamics
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
    def setPosition(self, x):
        self.params.x = x[0]
        self.params.y = x[1]
        self.params.theta = x[2]


    # Functions
    def cbf(self):
        distance = (1/self.params.ranges ** 2)
        derivDistance = (-2/self.params.ranges**3)
        return derivDistance + self.params.cbf_gamma0 * distance
    
    # Obtaining Lie derivatives of CBF
    def dcbf(self):
        jacobian_dcbf = jacobian(self.cbf)
        return jacobian_dcbf(self.x()).reshape((1,3))
    def lf_cbf(self):
        return self.dcbf().reshape((1,3)) @ self.f().reshape((3,1))
    def lg_cbf(self):
        return self.dcbf() @ self.g()
        
    # Constraints/Cost
    def constraints_cost(self):
        u_ref = np.zeros((self.params.udim, 1))

        B = self.cbf(self.x())
        LfB = float(self.lf_cbf())
        LgB = float(self.lg_cbf())

        A = np.array(([float(-LgB), 0]))
        b = np.array(([float(LfB) + float(self.params.cbfrate * B)]))
        
        # # u_min constraints 
        
        A = np.vstack((A, np.hstack(([np.eye(self.params.udim), np.zeros((self.params.udim, 1))]))))
        k = self.params.u_max * np.ones((self.params.udim, 1))
        b = np.vstack((b,k))

        # #u_max constraints
        A = np.vstack((A,np.hstack((-np.eye(self.params.udim), np.zeros((self.params.udim, 1))))))
        b = np.vstack((b,(-self.params.u_min * np.ones((self.params.udim, 1)))))

    # Cost
        

        weight_input = np.eye(self.params.udim)
        # cost = 0.5 [u' slack] H [u; slack] + f [u; slack]
        H = np.array(([float(weight_input), 0],[0, self.params.weightslack]))
        f_ = np.array(([float(weight_input) * float(u_ref)],[0]))
        
        try:
            x = solve_qp(H, f_, A, b, solver='clarabel')
            u = x[0]
            print("x")
            print(self.params.x)
            print("y")
            print(self.params.y)
            print("Theta")
            print(self.params.theta)
            print("U")
            print(u)
            #self.params.weightslack = float(x[1]) 
            return u
        except :
            print("error")
            return 0.0
        

        
