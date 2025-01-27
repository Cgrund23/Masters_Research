import sympy
import numpy as np
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
        self.x, self.y, self.gamma, self.theta, v = sympy.symbols(('x','y', 'gamma','theta','v'))


        f = [0 ,0 ,0 ,0]
        self.g = [[np.cos(self.theta), 0],[np.sin(self.theta),0],[np.tan(self.theta,1),0],[0,1]]

    def cbf(self, v , state):

        x = self.x
        x = x(1) 
        self.y = x(2) 
        self.theta = x(3)

        v = self.v
        xo = self.xo
        yo = self.yo
        d = self.d

        distance = (self.x - xo)^2 + (self.y - yo)^2 - d^2
        derivDistance = 2*(self.x-xo)*v*np.cos(self.theta) + 2*(self.y-yo)*v*np.sin(self.theta)
        cbf = derivDistance + self.cbf_gamma0 * distance 

    def ctrlCbfQp(obj, x, u_ref, verbose):              
                
        #tstart = tic
        B = obj.cbf(x)
        LfB = obj.lf_cbf(x)
        LgB = obj.lg_cbf(x)
            
        ## Constraints : A * u <= b
        # CBF constraint.
        A = [-LgB]
        b = [LfB + obj.params.cbf.rate * B]                
        # Add input constraints if u_max or u_min exists.
        A = [A eye(obj.udim)]
        b = [b obj.params.u_max]

        A = [A -eye(obj.udim)]
        b = [b -obj.params.u_min]


        ## Cost

        weight_input = eye(obj.udim)
        
        # cost = 0.5 u' H u + f u
        H = weight_input
        f_ = -weight_input * u_ref
        [u, ~, exitflag, ~] = quadprog(H, f_, A, b, [], [], [], [], [], options)