import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dst, idst
from scipy.linalg import solve
#from mcmc import Random_Walk
import sys
sys.path.append("../..") 
import cuqi

# the class dealing with FD for the Poisson equation on a staggered grid
# d/dx ( alpha du/dx ) = f
# u = 0. at x=0 and x=1
class poisson(cuqi.model.Model):
    """ Base cuqi model of the poisson problem"""
    def __init__(self, N=128, field_type="KL"):
        self.N = N # number of FD nodes
        self.dx = 1./self.N # step size
        self.Dx =  - np.diag( np.ones(N) , 0 ) + np.diag( np.ones(N-1) , 1 )
        vec = np.zeros(N)
        vec[0] = 1
        self.Dx = np.concatenate( [ vec.reshape( [1,-1] ), self.Dx], axis = 0 )
        self.Dx /= self.dx # FD derivative matrix

        x = np.linspace( self.dx,1.,N, endpoint=False )
        self.f = 10*np.exp( -( (x - 0.5)**2 ) / 0.02) # source term

        #self.alpha = gauss_field(self.N) # conductivity field
        domain_geometry = cuqi.geometry.KLField(N+1)
        range_geometry = cuqi.geometry.Continuous1D(N+1)

        super().__init__(self.forward, range_geometry, domain_geometry)

    # function computing observed function
    def solve_with_conductivity(self, true_alpha):
        Dxx = self.Dx.T@np.diag(true_alpha)@self.Dx
        u = solve(Dxx,self.f)
        return u

    # the forward map from field parameters to the solution
    def forward(self,p):
        a = np.exp( 10*self.domain_geometry.to_function(p) )
        Dxx = self.Dx.T@np.diag(a)@self.Dx
        return solve(Dxx,self.f)

if __name__ == "__main__":
    N = 128
    dx = np.pi/N
    x = np.linspace(dx/2,np.pi-dx/2,N+1)
    true_alpha = np.exp( 5*x*np.exp(-2*x)*np.sin(np.pi-x) )

    model = poisson(N=N)
    problem = poisson(N)
    y_obs = problem.solve_with_conductivity(true_alpha) 

    SNR = 100 # signal to noise ratio
    sigma = np.linalg.norm(y_obs)/SNR
    sigma2 = sigma*sigma # variance of the observation Gaussian noise

    likelihood = cuqi.distribution.Gaussian(model,sigma,np.eye(N+1))
    #%% Prior
    prior = cuqi.distribution.Gaussian(np.zeros((N+1,1)),1,np.eye(N+1))

    IP=cuqi.problem.BayesianProblem(likelihood,prior,y_obs)