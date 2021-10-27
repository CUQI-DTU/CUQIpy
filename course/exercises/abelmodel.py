import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dst, idst
from scipy.linalg import solve
#from mcmc import Random_Walk
import sys
sys.path.append("../..") 
import cuqi

# this class computes the sinogram of an image with rotational
# symmetry using the Abel integral operator that reduces the problem\
# to a 1D problem. The integral operator A uses numerical quadrature
# on off set s and t to avoid the singularity.
class abel(cuqi.model.Model):
    def __init__(self, N=1000):
        self.N = N # number of quadrature points
        self.h = 1./self.N # quadrature weight

        self.tvec = np.linspace( self.h/2, 1.-self.h/2, self.N ).reshape(1,-1) 
        svec = self.tvec.reshape(-1,1) + self.h/2

        tmat = np.tile( self.tvec, [self.N,1] )
        smat = np.tile( svec, [1,self.N] )
        
        idx = np.where(tmat<smat) # only applying the quadrature on 0<x<1

        self.A = np.zeros([self.N,self.N]) # Abel integral operator
        self.A[idx[0],idx[1]] = self.h/np.sqrt( np.abs( smat[idx[0],idx[1]] - tmat[idx[0],idx[1]] ) )

        #domain_geometry = cuqi.geometry.KLExpansion(self.tvec.reshape(-1))

        KL_map = lambda x: 10.*x
        grid = np.linspace( 0,1,self.N )
        domain_geometry = cuqi.geometry.MappedKL(grid, KL_map)
        range_geometry = cuqi.geometry.Continuous1D(self.N)

        super().__init__(self.forward, range_geometry, domain_geometry)

    def solve_with_image(self,x):
        return self.A@x

    # computing the sino gram
    def forward(self,p):
        im = self.domain_geometry.par2fun(p)
        return self.A@im
