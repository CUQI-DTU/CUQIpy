import numpy as np
from cuqi.geometry import _DefaultGeometry, Image2D, Continuous2D
from cuqi.operator import FirstOrderFiniteDifference
from cuqi.distribution import Distribution

class LMRF(Distribution):
    """Laplace distribution on the difference between neighboring nodes.

    For 1D `(physical_dim=1)`, the Laplace difference distribution assumes that

    .. math::

        x_i-x_{i-1} \sim \mathrm{Laplace}(0, b),

    where :math:`b` is the scale parameter.

    For 2D `(physical_dim=2)` the differences are defined in both horizontal and vertical directions.

    It is possible to define boundary conditions using the `bc_type` parameter.

    Parameters
    ----------
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
        prior = cuqi.distribution.LMRF(scale=0.1, dim=128)
 
    """
    def __init__(self, scale, bc_type="zero", physical_dim=None, **kwargs):
        # Init from abstract distribution class
        super().__init__(**kwargs)

        self.scale = scale
        self._bc_type = bc_type

        # Ensure geometry has shape
        if not self.geometry.fun_shape or self.geometry.par_dim == 1:
            raise ValueError(f"Distribution {self.__class__.__name__} must be initialized with geometry or dim greater than 1.")

        # Default physical_dim to geometry's dimension if not provided
        physical_dim = len(self.geometry.fun_shape) if physical_dim is None else physical_dim

        # Ensure provided physical dimension is either 1 or 2
        if physical_dim not in [1, 2]:
            raise ValueError("Only physical dimension 1 or 2 supported.")

        # Check for geometry mismatch
        if not isinstance(self.geometry, _DefaultGeometry) and physical_dim != len(self.geometry.fun_shape):
            raise ValueError(f"Specified physical dimension {physical_dim} does not match geometry's dimension {len(self.geometry.fun_shape)}")

        self._physical_dim = physical_dim

        # If physical_dim is 2 and geometry is _DefaultGeometry, replace it with Image2D
        if self._physical_dim == 2:
            N = int(np.sqrt(self.dim))
            num_nodes = (N, N)
            if isinstance(self.geometry, _DefaultGeometry):
                self.geometry = Image2D(num_nodes)
        else:  # self._physical_dim == 1
            num_nodes = self.dim

        self._diff_op = FirstOrderFiniteDifference(num_nodes=num_nodes, bc_type=bc_type)

    @property
    def location(self):
        return np.zeros(self.dim)

    def pdf(self, x):
        Dx = self._diff_op @ x  # np.diff(X)
        return (1/(2*self.scale))**(len(Dx)) * np.exp(-np.linalg.norm(Dx, ord=1, axis=0)/self.scale)

    def logpdf(self, x):
        Dx = self._diff_op @ x
        return len(Dx)*(-(np.log(2)+np.log(self.scale))) - np.linalg.norm(Dx, ord=1, axis=0)/self.scale

    def _sample(self,N=1,rng=None):
        raise NotImplementedError("'LMRF.sample' is not implemented. Sampling can be performed with the 'sampler' module.")

    # def cdf(self, x):   # TODO
    #     return 1/2 + 1/2*np.sign(x-self.loc)*(1-np.exp(-np.linalg.norm(x, ord=1, axis=0)/self.scale))

    # def sample(self):   # TODO
    #     p = np.random.rand(self.dim)
    #     return self.loc - self.scale*np.sign(p-1/2)*np.log(1-2*abs(p-1/2))
