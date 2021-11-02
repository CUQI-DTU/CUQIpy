import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse import hstack
from scipy.linalg import solve
from cuqi.samples import Samples
from cuqi.geometry import Geometry, StepExpansion, KLExpansion, CustomKL, Continuous1D, _DefaultGeometry

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
    
    def forward(self, x):
        # If input is samples then compute forward for each sample 
        # TODO: Check if this can be done all-at-once for computational speed-up
        if isinstance(x,Samples):
            Ns = x.samples.shape[-1]
            data_samples = np.zeros((self.range_dim,Ns))
            for s in range(Ns):
                data_samples[:,s] = self._forward_func(x.samples[:,s])
            return Samples(data_samples)
        else:
            return self._forward_func(x)

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
        return self._adjoint_func(y)

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
        

class Poisson_1D(Model):
    """ Base cuqi model of the Poisson 1D problem"""
    def __init__(self, N, L, source, field_type, skip=1, cov_fun=None, mean=None, std=None, d_KL=None, KL_map=lambda x: x):
        self.N = N-1          # number of FD nodes
        self.dx = 1./self.N   # step size
        self.Dx = - np.diag(np.ones(self.N), 0) + np.diag(np.ones(self.N-1), 1) 
        #
        vec = np.zeros(self.N)
        vec[0] = 1
        self.Dx = np.concatenate([vec.reshape([1,-1]), self.Dx], axis=0)
        self.Dx /= self.dx # FD derivative matrix

        # discretization
        self.x = np.linspace(0, L, self.N+1, endpoint=True)
        self.x_u = np.linspace(self.dx, L, self.N, endpoint=False)
        self.f = source(self.x_u)
        self.skip = skip
        #
        if field_type=="KL":
            domain_geometry = KLExpansion(self.x, mapping=KL_map)
        elif field_type=="CustomKL":
            domain_geometry = CustomKL(self.x, cov_fun, mean, std, d_KL, mapping=KL_map)
        elif field_type=="Step":
            domain_geometry = StepExpansion(self.x, mapping=KL_map)
        else:
            domain_geometry = Continuous1D(self.x, mapping=KL_map)
        range_geometry = Continuous1D(self.x_u)
        super().__init__(self.forward, range_geometry, domain_geometry)
    
    # forward projection: finite differences
    def forward(self, theta):
        kappa = self.domain_geometry.apply_map(theta)
        Dxx = self.Dx.T @ np.diag(kappa) @ self.Dx
        sol = solve(Dxx, self.f)
        return sol[::self.skip]

    def _solve_with_conductivity(self, true_kappa):
        Dxx = self.Dx.T@np.diag(true_kappa)@self.Dx
        u = solve(Dxx,self.f)
        return u[::self.skip]

    # compute gradient of target function 
    def gradient(self, func, kappa, eps=np.sqrt(np.finfo(np.float).eps)):
        return self.approx_jacobian(kappa, func, eps)
    
    # approximate the Jacobian matrix of callable function func
    def approx_jacobian(x, func, epsilon, *args):
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
    
    
class Heat_1D(Model):
    """ Base cuqi model of the Heat 1D problem"""
    def __init__(self, N, L, T, field_type, skip=1, cov_fun=None, mean=None, std=None, d_KL=None, KL_map=lambda x: x):
        self.N = N # number of discretization points
        self.dx = L/(self.N+1) # space step size
        #
        cfl = 5/11 # the cfl condition to have a stable solution
        self.dt = cfl*self.dx**2 # defining time step
        self.T = T # defining the maximum time
        self.MAX_ITER = int(self.T/self.dt) # number of time steps
        self.Dxx = np.diag( (1-2*cfl)*np.ones(self.N) ) + np.diag(cfl* np.ones(self.N-1),-1) + np.diag(cfl*np.ones(self.N-1),1) # FD diffusion operator

        # discretization
        self.x = np.linspace(self.dx, L, self.N, endpoint=False)
        self.skip = skip
        if field_type=="KL":
            domain_geometry = KLExpansion(self.x, mapping=KL_map)
        elif field_type=="CustomKL":
            domain_geometry = CustomKL(self.x, cov_fun, mean, std, d_KL, mapping=KL_map)
        elif field_type=="Step":
            domain_geometry = StepExpansion(self.x, mapping=KL_map)
        else:
            domain_geometry = Continuous1D(self.x, mapping=KL_map)
        range_geometry = Continuous1D(self.x)
        super().__init__(self.forward, range_geometry, domain_geometry)
    
    # computes the solution at t for a given initial condition
    # if makeplot is True, saves and displays all time steps
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
            (X,T) = np.meshgrid(self.x,T)

            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            ax.plot_surface(T,X,self.sol)
            plt.show()
        return u_old[::self.skip]

    # computes the solution at t for a given expansion coefficients
    def forward(self, theta):
        u0 = self.domain_geometry.apply_map(theta)
        sol = self._advance_time(u0)
        return sol[::self.skip]

    # compute gradient of target function 
    def gradient(self, func, kappa, eps=np.sqrt(np.finfo(np.float).eps)):
        return self.approx_jacobian(kappa, func, eps)
    
    # approximate the Jacobian matrix of callable function func
    def approx_jacobian(x, func, epsilon, *args):
        # x       - The state vector
        # func    - A vector-valued function of the form f(x,*args)
        # epsilon - The peturbation used to determine the partial derivatives
        x0 = np.asfarray(x)
        f0 = func(*((x0,)+args))
        jac = np.zeros([len(x0), len(f0)])
        dx = np.zeros(len(x0))
        for i in range(len(x0)):
            dx[i] = epsilon
            jac[i] = (func(*((x0+dx,)+args)) - f0)/epsilon
            dx[i] = 0.0
        return jac.transpose()
    
    
class Abel_1D(Model):
    """ Base cuqi model of the Abel 1D problem"""
    def __init__(self, N, L, field_type, cov_fun=None, mean=None, std=None, d_KL=None, KL_map=lambda x: x):
        self.N = N # number of quadrature points
        self.h = L/self.N # quadrature weight

        self.tvec = np.linspace(self.h/2, L-self.h/2, self.N).reshape(1, -1) 
        svec = self.tvec.reshape(-1, 1) + self.h/2
        tmat = np.tile( self.tvec, [self.N, 1] )
        smat = np.tile( svec, [1, self.N] )
        
        idx = np.where(tmat<smat) # only applying the quadrature on 0<x<1
        self.A = np.zeros([self.N,self.N]) # Abel integral operator
        self.A[idx[0], idx[1]] = self.h/np.sqrt( np.abs( smat[idx[0], idx[1]] - tmat[idx[0], idx[1]] ) )

        # discretization
        self.x = np.linspace(0, L, self.N)
        if field_type=="KL":
            domain_geometry = KLExpansion(self.x, mapping=KL_map)
        elif field_type=="CustomKL":
            domain_geometry = CustomKL(self.x, cov_fun, mean, std, d_KL, mapping=KL_map)
        elif field_type=="Step":
            domain_geometry = StepExpansion(self.x, mapping=KL_map)
        else:
            domain_geometry = Continuous1D(self.x, mapping=KL_map)
        range_geometry = Continuous1D(self.x)
        super().__init__(self.forward, range_geometry, domain_geometry)
    
    # forward model
    def forward(self, theta):
        u = self.domain_geometry.apply_map(theta)
        return self.A@u

    # compute gradient of target function 
    def gradient(self, kappa):
        return self.A.T@kappa
    
    
class Deconv_1D(Model):
    """ Base cuqi model of the deconvolution 1D problem"""
    def __init__(self, N, L, kernel, field_type, cov_fun=None, mean=None, std=None, d_KL=None, KL_map=lambda x: x):
        self.N = N # number of quadrature points
        self.h = L/self.N # quadrature weight
        self.x = np.linspace(0, L, self.N)
        
        # convolution matrix
        T1, T2 = np.meshgrid(self.x, self.x)
        A = self.h*kernel(T1, T2)
        maxval = A.max()
        A[A < 5e-3*maxval] = 0
        self.A = csc_matrix(A)   # make A sparse
        
        # discretization
        if field_type=="KL":
            domain_geometry = KLExpansion(self.x, mapping=KL_map)
        elif field_type=="CustomKL":
            domain_geometry = CustomKL(self.x, cov_fun, mean, std, d_KL, mapping=KL_map)
        elif field_type=="Step":
            domain_geometry = StepExpansion(self.x, mapping=KL_map)
        else:
            domain_geometry = Continuous1D(self.x, mapping=KL_map)
        range_geometry = Continuous1D(self.x)
        super().__init__(self.forward, range_geometry, domain_geometry)
    
    # forward model
    def forward(self, theta):
        u = self.domain_geometry.apply_map(theta)
        return self.A@u

    # compute gradient of target function 
    def gradient(self, kappa):
        return self.A.T@kappa