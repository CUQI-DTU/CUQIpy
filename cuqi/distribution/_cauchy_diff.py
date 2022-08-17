import numpy as np
import warnings
from cuqi.geometry import _DefaultGeometry, Image2D, _get_identity_geometries
from cuqi.distribution import Distribution
from cuqi.operator import FirstOrderFiniteDifference
from cuqi.geometry import _DefaultGeometry, Image2D, _get_identity_geometries

class Cauchy_diff(Distribution):
    """Cauchy distribution on the difference between neighboring nodes.

    Parameters
    ----------
    location : scalar or ndarray
        The location parameter of the distribution.

    scale : scalar
        The scale parameter of the distribution.

    bc_type : string
        The boundary conditions of the difference operator.

    physical_dim : int
        The physical dimension of what the distribution represents (can take the values 1 or 2).

    Example
    -------
    .. code-block:: python

        import cuqi
        import numpy as np
        prior = cuqi.distribution.Cauchy_diff(location=np.zeros(128), scale=0.1)

    Notes
    -----
    The pdf is given by

    .. math::

        \pi(\mathbf{x}) = \\frac{1}{(\pi\gamma)^n \left( 1+\left( \\frac{\mathbf{D}(\mathbf{x}-\mathbf{x}_0)}{\gamma} \\right)^2 \\right) },

    where :math:`\mathbf{x}_0\in \mathbb{R}^n` is the location parameter, :math:`\gamma` is the scale, :math:`\mathbf{D}` is the difference operator.
 
    """
   
    def __init__(self, location, scale, bc_type="zero", physical_dim=1, **kwargs):
        # Init from abstract distribution class
        super().__init__(**kwargs) 
        
        self.location = location
        self.scale = scale
        self._bc_type = bc_type
        self._physical_dim = physical_dim

        if physical_dim == 2:
            N = int(np.sqrt(self.dim))
            num_nodes = (N, N)
            if isinstance(self.geometry, _DefaultGeometry):
                self.geometry = Image2D(num_nodes)
            print("Warning: 2D Cauchy_diff is still experimental. Use at own risk.")
        elif physical_dim == 1:
            num_nodes = self.dim
        else:
            raise ValueError("Only physical dimension 1 or 2 supported.")

        self._diff_op = FirstOrderFiniteDifference(num_nodes=num_nodes, bc_type=bc_type)

    def logpdf(self, x):
        Dx = self._diff_op @ (x-self.location)
        # g_logpr = (-2*Dx/(Dx**2 + gamma**2)) @ D
        return -len(Dx)*np.log(np.pi) + sum(np.log(self.scale) - np.log(Dx**2 + self.scale**2))
    
    def gradient(self, val, **kwargs):
        #Avoid complicated geometries that change the gradient.
        if not type(self.geometry) in _get_identity_geometries():
            raise NotImplementedError("Gradient not implemented for distribution {} with geometry {}".format(self,self.geometry))

        if not callable(self.location): # for prior
            diff = self._diff_op._matrix @ val
            return (-2*diff/(diff**2+self.scale**2)) @ self._diff_op._matrix
        else:
            warnings.warn('Gradient not implemented for {}'.format(type(self.location)))

    def _sample(self,N=1,rng=None):
        raise NotImplementedError("'Cauchy_diff.sample' is not implemented. Sampling can be performed with the 'sampler' module.")

    # def cdf(self, x):   # TODO
    #     return 1/np.pi * np.atan((x-self.loc)/self.scale)

    # def sample(self):   # TODO
    #     return self.loc + self.scale*np.tan(np.pi*(np.random.rand(self.dim)-1/2))
