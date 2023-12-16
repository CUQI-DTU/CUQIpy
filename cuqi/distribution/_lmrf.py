import numpy as np
from cuqi.geometry import _DefaultGeometry1D, Image2D
from cuqi.operator import FirstOrderFiniteDifference
from cuqi.distribution import Distribution
from cuqi.utilities import force_ndarray

class LMRF(Distribution):
    """Laplace distribution on the difference between neighboring nodes.

    For 1D, the Laplace difference distribution assumes that

    .. math::

        x_i-x_{i-1} \sim \mathrm{Laplace}(0, b),

    where :math:`b` is the scale parameter.

    For 2D the differences are defined in both horizontal and vertical directions.

    It is possible to define boundary conditions using the `bc_type` parameter.

    The location parameter is a shift of the :math:`\mathbf{x}`.

    Parameters
    ----------
    location : scalar or ndarray
        The location parameter of the distribution.

    scale : scalar
        The scale parameter of the distribution.

    bc_type : string
        The boundary conditions of the difference operator.

    Example
    -------
    .. code-block:: python

        import cuqi
        prior = cuqi.distribution.LMRF(location=0, scale=0.1, geometry=128)
 
    """
    def __init__(self, location=None, scale=None, bc_type="zero", **kwargs):
        # Init from abstract distribution class
        super().__init__(**kwargs)

        self.location = location
        self.scale = scale
        self._bc_type = bc_type

        # Ensure geometry has shape
        if not self.geometry.fun_shape or self.geometry.par_dim == 1:
            raise ValueError(f"Distribution {self.__class__.__name__} must be initialized with supported geometry (geometry of which the fun_shape is not None) and has parameter dimension greater than 1.")

        # Default physical_dim to geometry's dimension if not provided
        physical_dim = len(self.geometry.fun_shape)

        # Ensure provided physical dimension is either 1 or 2
        if physical_dim not in [1, 2]:
            raise ValueError("Only physical dimension 1 or 2 supported.")

        self._physical_dim = physical_dim

        if self._physical_dim == 2:
            N = int(np.sqrt(self.dim))
            num_nodes = (N, N)
        else: 
            num_nodes = self.dim

        self._diff_op = FirstOrderFiniteDifference(num_nodes=num_nodes, bc_type=bc_type)

    @property
    def location(self):
        return self._location
    
    @location.setter
    def location(self, value):
        self._location = force_ndarray(value, flatten=True)

    def pdf(self, x):
        Dx = self._diff_op @ (x-self.location)  # np.diff(X)
        return (1/(2*self.scale))**(len(Dx)) * np.exp(-np.linalg.norm(Dx, ord=1, axis=0)/self.scale)

    def logpdf(self, x):
        Dx = self._diff_op @ (x-self.location)
        return len(Dx)*(-(np.log(2)+np.log(self.scale))) - np.linalg.norm(Dx, ord=1, axis=0)/self.scale

    def _sample(self,N=1,rng=None):
        raise NotImplementedError("'LMRF.sample' is not implemented. Sampling can be performed with the 'sampler' module.")

    # def cdf(self, x):   # TODO
    #     return 1/2 + 1/2*np.sign(x-self.loc)*(1-np.exp(-np.linalg.norm(x, ord=1, axis=0)/self.scale))

    # def sample(self):   # TODO
    #     p = np.random.rand(self.dim)
    #     return self.loc - self.scale*np.sign(p-1/2)*np.log(1-2*abs(p-1/2))
