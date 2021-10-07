import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dst, idst
import cuqi

# this class solves the heat equation with FD method
# d^2/dx^2 u = f,  x in [0,pi]
# u(0,t) = u(pi,t) = 0
# u(x,0) = u0
class heat(cuqi.model.Model):
    """ Base cuqi model of the heat problem"""
    def __init__(self, N=128, field_type="KL"):

        self.N = N # number of discretization points
        self.dx = np.pi/(self.N+1) # space step size

        self.grid = np.linspace(self.dx,np.pi,self.N,endpoint=False)

        cfl= 5/11 # the cfl condition to have a stable solution
        self.dt = cfl*self.dx**2 # defining time step
        self.T = .2 # defining the maximum time
        self.MAX_ITER = int(self.T/self.dt) # number of time steps

        self.Dxx = np.diag( (1-2*cfl)*np.ones(self.N) ) + np.diag(cfl* np.ones(self.N-1),-1) + np.diag(cfl*np.ones(self.N-1),1) # FD diffusion operator

        if field_type=="KL":
            domain_geometry = cuqi.geometry.KLExpansion(self.grid)
        elif field_type=="Step":
            domain_geometry = cuqi.geometry.StepExpansion(self.grid)
        range_geometry = cuqi.geometry.Continuous1D(N)

        super().__init__(self.forward, range_geometry, domain_geometry)

    # computes the solution at t=0.2 for a given initial condition
    # if makeplot is True, saves and displays all time steps.
    def _advance_time(self, u0, makeplot=False):
        u_old = np.copy(u0)
        
        if makeplot:
            self.sol = [ np.copy(u0) ]
        
        for i in range(self.MAX_ITER):
            u_old = self.Dxx@u_old
            if makeplot:
                self.sol.append(u_old)
        
        if makeplot:
            self.sol = np.array(self.sol)

            T = np.array( range(0,self.MAX_ITER+1) )
            (X,T) = np.meshgrid(self.grid,T)

            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            ax.plot_surface(T,X,self.sol)
            plt.show()

        return u_old   

    # computes the solution at t=0.2 for a given expansion coefficients
    def forward(self,p):
        u0 = self.domain_geometry.to_function(p)
        return self._advance_time(u0)