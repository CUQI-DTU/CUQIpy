import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse import hstack

from cuqi.samples import Samples

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
        if isinstance(x,Samples):
            Ns = x.samples.shape[-1]
            data_samples = np.zeros((self.dim[0],Ns))
            for s in range(Ns):
                data_samples[:,s] = self.forward(x.samples[:,s])
            return Samples(data_samples)
        else:
            return self._forward_func(x)
    
class LinearModel(Model):
    # Linear forward model with forward and adjoint (transpose).
    
    def __init__(self,forward,adjoint=None,dim=None):
        
        #Assume forward is matrix if not callable (TODO: add more checks)
        if not callable(forward): 
            forward_func = lambda x: self._matrix@x
            adjoint_func = lambda y: self._matrix.T@y
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

    def adjoint(self,y):
        return self._adjoint_func(y)

    def get_matrix(self):
        if self._matrix is not None: #Matrix exists so return it
            return self._matrix
        else:
            #TODO: Can we compute this faster while still in sparse format?
            mat = csc_matrix((self.dim[0],0)) #Sparse (m x 1 matrix)
            e = np.zeros(self.dim[1])
            
            # Stacks sparse matricies on csc matrix
            for i in range(self.dim[1]):
                e[i] = 1
                col_vec = self.forward(e)
                mat = hstack((mat,col_vec[:,None])) #mat[:,i] = self.forward(e)
                e[i] = 0

            #Store matrix for future use
            self._matrix = mat

            return self._matrix

    def __mul__(self, x):
        return self.forward(x)
    
    def __matmul__(self, x):
        return self*x

        
