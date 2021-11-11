import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse import hstack
from scipy.linalg import solve
from cuqi.samples import Samples, CUQIarray
from cuqi.geometry import Geometry, StepExpansion, KLExpansion, CustomKL, Continuous1D, _DefaultGeometry
import cuqi
import matplotlib.pyplot as plt

class Model(object):
    """Generic model defined by a forward operator.

    Parameters
    -----------
    forward : 2D ndarray or callable function.
        Forward operator.

    range_geometry : integer or cuqi.geometry.Geometry
        If integer is given a _DefaultGeometry is created with dimension of the integer.

    domain_geometry : integer or cuqi.geometry.Geometry
        If integer is given a _DefaultGeometry is created with dimension of the integer.

    Attributes
    -----------
    range_geometry : cuqi.geometry.Geometry
        The geometry representing the range.

    domain_geometry : cuqi.geometry.Geometry
        The geometry representing the domain.

    Methods
    -----------
    :meth:`forward` the forward operator.
    :meth:`range_dim` the dimension of the range.
    :meth:`domain_dim` the dimension of the domain.
    """
    def __init__(self, forward, range_geometry, domain_geometry):

        #Check if input is callable
        if callable(forward) is not True:
            raise TypeError("Forward needs to be callable function of some kind")
            
        #Store forward func
        self._forward_func = forward
         
        #Store range_geometry
        if isinstance(range_geometry, int):
            self.range_geometry = _DefaultGeometry(grid=range_geometry)
        elif isinstance(range_geometry, Geometry):
            self.range_geometry = range_geometry
        elif range_geometry is None:
            raise AttributeError("The parameter 'range_geometry' is not specified by the user and it connot be inferred from the attribute 'forward'.")
        else:
            raise TypeError("The parameter 'range_geometry' should be of type 'int' or 'cuqi.geometry.Geometry'.")

        #Store domain_geometry
        if isinstance(domain_geometry, int):
            self.domain_geometry = _DefaultGeometry(grid=domain_geometry)
        elif isinstance(domain_geometry, Geometry):
            self.domain_geometry = domain_geometry
        elif domain_geometry is None:
            raise AttributeError("The parameter 'domain_geometry' is not specified by the user and it connot be inferred from the attribute 'forward'.")
        else:
            raise TypeError("The parameter 'domain_geometry' should be of type 'int' or 'cuqi.geometry.Geometry'.")

    @property
    def domain_dim(self): 
        return self.domain_geometry.dim

    @property
    def range_dim(self): 
        return self.range_geometry.dim
    
    def forward(self, x, is_par=True):
        # If input is samples then compute forward for each sample 
        # TODO: Check if this can be done all-at-once for computational speed-up

        if type(x) is CUQIarray:
            x = x.funvals
        else:
            if is_par:
                x = self.domain_geometry.apply_map(x)

        if isinstance(x,Samples):
            Ns = x.samples.shape[-1]
            data_samples = np.zeros((self.range_dim,Ns))
            for s in range(Ns):
                data_samples[:,s] = self._forward_func(x.samples[:,s])
            return Samples(data_samples)
        else:
            out = self._forward_func(x)
            if type(x) is CUQIarray:
                out = CUQIarray(out, geometry=self.range_geometry)
            return out

    def __call__(self,x):
        return self.forward(x)

    def gradient(self,x):
        raise NotImplementedError("Gradient is not implemented for this model.")
    
    def __len__(self):
        return self.range_dim
    
class LinearModel(Model):
    """Model based on a Linear forward operator.

    Parameters
    -----------
    forward : 2D ndarray or callable function.
        Forward operator.

    adjoint : 2d ndarray or callable function. (optional if matrix is passed as forward)

    range_geometry : integer or cuqi.geometry.Geometry (optional)
        If integer is given a _DefaultGeometry is created with dimension of the integer.

    domain_geometry : integer or cuqi.geometry.Geometry (optional)
        If integer is given a _DefaultGeometry is created with dimension of the integer.

    Attributes
    -----------
    range_geometry : cuqi.geometry.Geometry
        The geometry representing the range.

    domain_geometry : cuqi.geometry.Geometry
        The geometry representing the domain.

    Methods
    -----------
    :meth:`forward` the forward operator.
    :meth:`range_dim` the dimension of the range.
    :meth:`domain_dim` the dimension of the domain.
    :meth:`get_matrix` returns an ndarray with the matrix representing the forward operator.
    """
    # Linear forward model with forward and adjoint (transpose).
    
    def __init__(self,forward,adjoint=None,range_geometry=None,domain_geometry=None):
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

        # Use matrix to derive range_geometry and domain_geometry
        if matrix is not None:
            if range_geometry is None:
                range_geometry = _DefaultGeometry(grid=matrix.shape[0])
            if domain_geometry is None:
                domain_geometry = _DefaultGeometry(grid=matrix.shape[1])  

        #Initialize Model class
        super().__init__(forward_func,range_geometry,domain_geometry)

        #Add adjoint
        self._adjoint_func = adjoint_func

        #Store matrix privately
        self._matrix = matrix

        if matrix is not None: 
            assert(self.range_dim  == matrix.shape[0]), "The parameter 'forward' dimensions are inconsistent with the parameter 'range_geometry'"
            assert(self.domain_dim == matrix.shape[1]), "The parameter 'forward' dimensions are inconsistent with parameter 'domain_geometry'"

    def adjoint(self,y):
        out = self._adjoint_func(y)
        if type(y) is CUQIarray:
            out = CUQIarray(out, is_par=False, geometry=self.domain_geometry)
        return out


    def get_matrix(self):
        if self._matrix is not None: #Matrix exists so return it
            return self._matrix
        else:
            #TODO: Can we compute this faster while still in sparse format?
            mat = csc_matrix((self.range_dim,0)) #Sparse (m x 1 matrix)
            e = np.zeros(self.domain_dim)
            
            # Stacks sparse matrices on csc matrix
            for i in range(self.domain_dim):
                e[i] = 1
                col_vec = self.forward(e)
                mat = hstack((mat,col_vec[:,None])) #mat[:,i] = self.forward(e)
                e[i] = 0

            #Store matrix for future use
            self._matrix = mat

            return self._matrix

    def gradient(self,x):
        """Evaluate the gradient of the forward map with respect to the model input."""
        return self.adjoint(x)
    
    def __matmul__(self, x):
        return self.forward(x)

    @property
    def T(self):
        """Transpose of linear model. Returns a new linear model acting as the transpose."""
        transpose = LinearModel(self.adjoint,self.forward,self.domain_geometry,self.range_geometry)
        if self._matrix is not None:
            transpose._matrix = self._matrix.T
        return transpose
        

class PDEModel(Model):

    def __init__(self, PDE, range_geometry, domain_geometry):
        #....
        if not isinstance(PDE,cuqi.pde.PDE):
            raise ValueError("PDE needs to be a cuqi PDE.")

        super().__init__(self._forward_func, range_geometry, domain_geometry)

        self.pde = PDE

    def _forward_func(self,x):
        
        self.pde.assemble(parameter=x)

        sol = self.pde.solve()

        obs = self.pde.observe(sol)

        return obs

    # compute gradient of target function 
    def gradient(self, func, kappa, eps=np.sqrt(np.finfo(np.float).eps)):
        return self._approx_jacobian(kappa, func, eps)
    
    # approximate the Jacobian matrix of callable function func
    def _approx_jacobian(x, func, epsilon, *args):
        # x       - The state vector
        # func    - A vector-valued function of the form f(x,*args)
        # epsilon - The peturbation used to determine the partial derivatives
        # The approximation is done using forward differences
        x0 = np.asfarray(x)
        f0 = func(*((x0,)+args))
        jac = np.zeros([len(x0), len(f0)])
        dx = np.zeros(len(x0))
        for i in range(len(x0)):
            dx[i] = epsilon
            jac[i] = (func(*((x0+dx,)+args)) - f0)/epsilon
            dx[i] = 0.0
        return jac.transpose()
        