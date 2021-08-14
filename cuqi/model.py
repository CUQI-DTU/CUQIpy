import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse import hstack
from cuqi.samples import Samples
from cuqi.geometry import Geometry, Continuous1D
import warnings

class Model(object):
    """
    Parameters
    ----------
    forward : 2D ndarray or callable function
        Forward operator
    """
    def __init__(self,forward,range=None,domain=None):
        """
        Parameters
        ----------
        forward : 2D ndarray or callable function
            Forward operator
        """
        #Check if input is callable
        if callable(forward) is not True:
            raise TypeError("Forward needs to be callable function of some kind")
            
        #Store forward func
        self._forward_func = forward
        
        # dim
        self._dim = None 

        #Store range
        if isinstance(range, int):
            self.rangeGeom = Continuous1D(dim=[range])
        elif isinstance(range, Geometry):
            self.rangeGeom = range
        elif range is None:
            raise AttributeError("The parameter 'range' is not specified by the user and it connot be inferred from the attribute 'forward'.")
        else:
            raise TypeError("The parameter 'range' should be of type 'int' or 'cuqi.geometry.Geometry'.")

        #Store domain
        if isinstance(domain, int):
            self.domainGeom = Continuous1D(dim=[domain])
        elif isinstance(domain, Geometry):
            self.domainGeom = domain
        elif domain is None:
            raise AttributeError("The parameter 'domain' is not specified by the user and it connot be inferred from the attribute 'forward'.")
        else:
            raise TypeError("The parameter 'domain' should be of type 'int' or 'cuqi.geometry.Geometry'.")

    @property
    def dim(self): #dim is derived from rangeGeom and domainGeom objects
        dim_old = self._dim 
        if self.rangeGeom is not None and self.domainGeom is not None: 
            self._dim = (len(self.rangeGeom.grid.flat), len(self.domainGeom.grid.flat)) #TODO: change len(self.domainGeom.grid) to self.domainGeom.ndofs
        if dim_old is not None and self._dim != dim_old:
            warnings.warn("'Model.dim' value was changed to be compatible with 'rangeGeom' and 'domainGeom' ")
        return self._dim
               
    def forward(self, x):
        # If input is samples then compute forward for each sample 
        # TODO: Check if this can be done all-at-once for computational speed-up
        if isinstance(x,Samples):
            Ns = x.samples.shape[-1]
            data_samples = np.zeros((self.dim[0],Ns))
            for s in range(Ns):
                data_samples[:,s] = self._forward_func(x.samples[:,s])
            return Samples(data_samples)
        else:
            return self._forward_func(x)
    
class LinearModel(Model):
    """
    Class-based representation of Linear forward operator.

    :param forward: A matrix or callable function representing forward operator.
    :param adjoint: A callable function representing adjoint operator.
    :param dim: Dimensions of linear model.
    """
    # Linear forward model with forward and adjoint (transpose).
    
    def __init__(self,forward,adjoint=None,range=None,domain=None):
        #Assume forward is matrix if not callable (TODO: add more checks)
        if not callable(forward): 
            forward_func = lambda x: self._matrix@x
            adjoint_func = lambda y: self._matrix.T@y
            matrix = forward
        else:
            forward_func = forward
            adjoint_func = adjoint
            matrix = None

        #Check if input is callable
        if callable(adjoint_func) is not True:
            raise TypeError("Adjoint needs to be callable function of some kind")

        #Add adjoint
        self._adjoint_func = adjoint_func

        #Store matrix privately
        self._matrix = matrix

        # Use matrix to derive range and domain
        if matrix is not None:
            if range is None:
                range = Continuous1D(dim=[matrix.shape[0]])
            if domain is None:
                domain = Continuous1D(dim=[matrix.shape[1]])  

        #Initialize Model class
        super().__init__(forward_func,range,domain)

        if matrix is not None: 
            assert(len(self.rangeGeom.grid.flat)  == matrix.shape[0]), "The parameter 'forward' dimensions are inconsistent with the parameter 'range'"
            assert(len(self.domainGeom.grid.flat)  == matrix.shape[1]), "The parameter 'forward' dimensions are inconsistent with parameter 'range'"

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

        
