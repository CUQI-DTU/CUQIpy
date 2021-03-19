import numpy as np

class Model(object):
    # Generic Model class.
    
    def __init__(self,forward,dim=[]):
        
        #Check if input is callable
        if callable(forward) is not True:
            raise TypeError("Forward needs to be callable function of some kind")
            
        #Store forward func
        self.forward_func = forward
        
        #Store dimension
        self.dim = dim
               
    def forward(self, x):
        return self.forward_func(x)
    
class LinearModel(Model):
    # Linear forward model with forward and adjoint (transpose).
    # Only accepts callable functions for now.
    
    def __init__(self,forward,adjoint,dim=[]):
        
        #Initialize Model class
        Model.__init__(self,forward,dim)
        
        #Check if input is callable
        if callable(adjoint) is not True:
            raise TypeError("Adjoint needs to be callable function of some kind")
            
        #Add adjoint
        self.adjoint = adjoint
        
    def get_matrix(self):
        mat = np.zeros((self.dim[0],self.dim[1]))
        e = np.zeros(self.dim[1])
        
        for i in range(self.dim[1]):
            e[i] = 1
            mat[:,i] = self.forward(e)
            e[i] = 0
            
        return mat
        
        
