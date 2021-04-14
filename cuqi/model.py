import numpy as np
from scipy.sparse import csc_matrix

class Model(object):
    # Generic Model class.
    
    def __init__(self,forward,dim=[]):
        
        #Check if input is callable
        if callable(forward) is not True:
            raise TypeError("Forward needs to be callable function of some kind")
            
        #Store forward func
        self._forward_func = forward
        
        #Store dimension
        self.dim = dim
               
    def forward(self, x):
        return self._forward_func(x)
    
class LinearModel(Model):
    # Linear forward model with forward and adjoint (transpose).
    
    def __init__(self,forward,adjoint=None,dim=None):
        
        #Assume forward is matrix if not callable (TODO: add more checks)
        if not callable(forward): 
            forward_func = lambda x: forward@x
            adjoint_func = lambda y: forward.T@y
            matrix = forward
            dim = forward.shape
        else:
            forward_func = forward
            adjoint_func = adjoint
            matrix = None
            dim = dim

        #Initialize Model class
        super().__init__(forward_func,dim)
        
        #Check if input is callable
        if callable(adjoint_func) is not True:
            raise TypeError("Adjoint needs to be callable function of some kind")
            
        #Add adjoint
        self._adjoint_func = adjoint_func

        #Store matrix privately
        self._matrix = matrix

    def adjoint(self):
        return self._adjoint_func

    def get_matrix(self):
        if self._matrix is not None: #Matrix exists so return it
            return self._matrix
        else:
            mat = csc_matrix((self.dim[0],self.dim[1])) #Sparse matrix
            e = np.zeros(self.dim[1])
            
            for i in range(self.dim[1]):
                e[i] = 1
                mat[:,i] = self.forward(e)
                e[i] = 0

            #Store matrix for future use
            self._matrix = mat

            return self._matrix
        
