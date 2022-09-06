import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse import linalg as splinalg
from scipy.linalg import dft
from cuqi.geometry import _DefaultGeometry, Image2D, _get_identity_geometries
from cuqi.utilities import sparse_cholesky
from cuqi import config
from cuqi.operator import PrecisionFiniteDifference
from cuqi.distribution import Distribution

class GMRF(Distribution):
    """ Gaussian Markov random field (GMRF).
       
    Parameters
    ----------
    mean : array_like
        Mean of the GMRF. 
        
    prec : float
        Precision of the GMRF.

    physical_dim : int
        The physical dimension of what the distribution represents (can take the values 1 or 2).

    bc_type : str
        The type of boundary conditions to use. Can be 'zero', 'periodic' or 'neumann'.

    order : int
        The order of the GMRF. Can be 0, 1 or 2.

    Notes
    -----
    The GMRF defines a distribution over a set of points where each point conditioned on all the others follow a Gaussian distribution.

    For 1D `(physical_dim=1)`, the current implementation provides three different cases:

    * Order 0: :math:`x_i \sim \mathcal{N}(\mu_i, \delta^{-1})`,
    * Order 1: :math:`x_i \mid x_{i-1},x_{i+1} \sim \mathcal{N}(\mu_i+(x_{i-1}+x_{i+1})/2, (2\delta)^{-1}))`,
    * Order 2: :math:`x_i \mid x_{i-1},x_{i+1} \sim \mathcal{N}(\mu_i+(-x_{i-1}+2x_i-x_{i+1})/4, (4\delta)^{-1}))`,

    where :math:`\delta` is the `prec` parameter and the `mean` parameter is the mean :math:`\mu_i` for each :math:`i`.

    For 2D `(physical_dim=2)`, order 0, 1, and 2 are also supported in which the differences are defined in both horizontal and vertical directions.

    It is possible to define boundary conditions for the GMRF using the `bc_type` parameter.

    **Illustration as a Gaussian distribution**

    It may be beneficial to illustrate the GMRF distribution for a specific parameter setting. In 1D with zero boundary conditions,
    the GMRF distribution can be represented by a Gaussian, :math:`\mathcal{N}(\mu, \mathbf{P}^{-1})`, with mean :math:`\mu` and the following precision matrices depending on the order:

    * Order 0:

    .. math::

        \mathbf{P} = \delta \mathbf{I}.

    * Order 1: 

    .. math::
    
        \mathbf{P} = \delta 
        \\begin{bmatrix} 
             2  & -1        &           &           \\newline
            -1  &  2        & -1        &           \\newline
                & \ddots    & \ddots    & \ddots    \\newline
                &           & -1        & 2         
        \end{bmatrix}.

    * Order 2:

    .. math::

        \mathbf{P} = \delta
        \\begin{bmatrix}
             6   & -4       &  1        &           &           &           \\newline
            -4   &  6       & -4        & 1         &           &           \\newline
            1    & -4       &  6        & -4        & 1         &           \\newline
                 & \ddots   & \ddots    & \ddots    & \ddots    & \ddots    \\newline
                 &          & \ddots    & \ddots    & \ddots    & \ddots    \\newline
                 &          &           & 1         & -4        &  6        \\newline
        \end{bmatrix}.

    **General representation**

    In general we can define the GMRF distribution on each point by

    .. math::

        x_i \mid \mathbf{x}_{\partial_i} \sim \mathcal{N}\left(\sum_{j \in \partial_i} \\beta_{ij} x_j, \kappa_i^{-1}\\right),

    where :math:`\kappa_i` is the precision of each Gaussian and :math:`\\beta_{ij}` are coefficients defining the structure of the GMRF.

    For more details see: See Bardsley, J. (2018). Computational Uncertainty Quantification for Inverse Problems, Chapter 4.2.

    """
        
    def __init__(self, mean, prec, physical_dim=1, bc_type='zero', order=1, is_symmetric=True, **kwargs): 
        super().__init__(is_symmetric=is_symmetric, **kwargs) #TODO: This calls Distribution __init__, should be replaced by calling Gaussian.__init__ 

        self.mean = mean.reshape(len(mean), 1)
        self.prec = prec
        self._partition_size = int(len(mean)**(1/physical_dim))
        self._bc_type = bc_type      # boundary conditions
        self._physical_dim = physical_dim
        
        num_nodes = tuple(self._partition_size for _ in range(physical_dim))
        if physical_dim == 2: #TODO. Remove once _DefaultGeometry is implemented for 2D.
            if isinstance(self.geometry, _DefaultGeometry):
                self.geometry = Image2D(num_nodes)

        self._prec_op = PrecisionFiniteDifference(num_nodes, bc_type=bc_type, order=order) 
        self._diff_op = self._prec_op._diff_op      
                   
        # compute Cholesky and det
        if (bc_type == 'zero'):    # only for PSD matrices
            self._rank = self.dim
            self._chol = sparse_cholesky(self._prec_op.get_matrix()).T
            self._logdet = 2*sum(np.log(self._chol.diagonal()))
            # L_cholmod = cholesky(self.L, ordering_method='natural')
            # self.chol = L_cholmod
            # self.logdet = L_cholmod.logdet()
            # 
            # np.log(np.linalg.det(self.L.todense()))
        elif (bc_type == 'periodic') or (bc_type == 'neumann'):
            # Print warning that periodic and Neumann boundary conditions are experimental
            print("Warning (GMRF): Periodic and Neumann boundary conditions are experimental. Sampling using Linear_RTO may not produce fully accurate results.")

            eps = np.finfo(float).eps
            self._rank = self.dim - 1   #np.linalg.matrix_rank(self.L.todense())
            self._chol = sparse_cholesky(self._prec_op + np.sqrt(eps)*eye(self.dim, dtype=int)).T
            if (self.dim > config.MAX_DIM_INV):  # approximate to avoid 'excesive' time
                self._logdet = 2*sum(np.log(self._chol.diagonal()))
            else:
                # eigval = eigvalsh(self.L.todense())
                self._L_eigval = splinalg.eigsh(self._prec_op.get_matrix(), self._rank, which='LM', return_eigenvectors=False)
                self._logdet = sum(np.log(self._L_eigval))
        else:
            raise ValueError('bc_type must be "zero", "periodic" or "neumann"')

    @property
    def prec(self):
        return self._prec

    @prec.setter
    def prec(self, value):
        # We store the precision as a scalar to match existing code in this class,
        # but allow user and other code to provide it as a 1D ndarray with 1 element.
        if isinstance(value, np.ndarray):
            if len(value) == 1:
                value = value[0]
            else:
                raise ValueError('Precision must be a scalar or a 1D array with a single scalar element.')
        self._prec = value

    @property 
    def dim(self):  
        if self._physical_dim == 1:
            return self._partition_size 
        elif self._physical_dim==2:
            return self._partition_size**2
        raise ValueError("attribute dom can be either 1 or 2")

    def logpdf(self, x):
        mean = self.mean.flatten()
        const = 0.5*(self._rank*(np.log(self.prec)-np.log(2*np.pi)) + self._logdet)
        y = const - 0.5*( self.prec*((x-mean).T @ (self._prec_op @ (x-mean))) )
        # = sps.multivariate_normal.logpdf(x.T, self.mean.flatten(), np.linalg.inv(self.prec*self.L.todense()))
        return y

    def pdf(self, x):
        # = sps.multivariate_normal.pdf(x.T, self.mean.flatten(), np.linalg.inv(self.prec*self.L.todense()))
        return np.exp(self.logpdf(x))

    def gradient(self, x):
        #Avoid complicated geometries that change the gradient.
        if not type(self.geometry) in _get_identity_geometries():
            raise NotImplementedError("Gradient not implemented for distribution {} with geometry {}".format(self,self.geometry))

        if not callable(self.mean):
            mean = self.mean.flatten()
            return -(self.prec*self._prec_op) @ (x-mean)

    def _sample(self, N=1, rng=None):
        if (self._bc_type == 'zero'):

            if rng is not None:
                xi = rng.standard_normal((self.dim, N))   # standard Gaussian
            else:
                xi = np.random.randn(self.dim, N)   # standard Gaussian

            if N == 1:
                s = self.mean.flatten() + (1/np.sqrt(self.prec))*splinalg.spsolve(self._chol.T, xi)
            else:
                s = self.mean + (1/np.sqrt(self.prec))*splinalg.spsolve(self._chol.T, xi)
            # s = self.mean + (1/np.sqrt(self.prec))*L_cholmod.solve_Lt(xi, use_LDLt_decomposition=False) 
                        
        elif (self._bc_type == 'periodic'):
            
            if self._physical_dim == 2:
                raise NotImplementedError("Sampling not implemented for periodic boundary conditions in 2D")

            if rng is not None:
                xi = rng.standard_normal((self.dim, N)) + 1j*rng.standard_normal((self.dim, N))
            else:
                xi = np.random.randn(self.dim, N) + 1j*np.random.randn(self.dim, N)
            
            F = dft(self.dim, scale='sqrtn')   # unitary DFT matrix
            # eigv = eigvalsh(self.L.todense()) # splinalg.eigsh(self.L, self.rank, return_eigenvectors=False)           
            eigv = np.hstack([self._L_eigval, self._L_eigval[-1]])  # repeat last eigval to complete dim
            L_sqrt = diags(np.sqrt(eigv)) 
            s = self.mean + (1/np.sqrt(self.prec))*np.real(F.conj() @ splinalg.spsolve(L_sqrt, xi))
            # L_sqrt = pinvh(np.diag(np.sqrt(eigv)))
            # s = self.mean + (1/np.sqrt(self.prec))*np.real(F.conj() @ (L_sqrt @ xi))
            
        elif (self._bc_type == 'neumann'):

            if rng is not None:
                xi = rng.standard_normal((self._diff_op.shape[0], N))   # standard Gaussian
            else:
                xi = np.random.randn(self._diff_op.shape[0], N)   # standard Gaussian

            s = self.mean + (1/np.sqrt(self.prec))* \
                splinalg.spsolve(self._chol.T, (splinalg.spsolve(self._chol, (self._diff_op.T @ xi)))) 
        else:
            raise TypeError('Unexpected BC type (choose from zero, periodic, neumann or none)')

        return s
    
    @property
    def sqrtprec(self):
        return np.sqrt(self.prec)*self._chol.T

    @property
    def sqrtprecTimesMean(self):
        return (self.sqrtprec@self.mean).flatten()
        