import numpy as np
from cuqi.geometry import _DefaultGeometry, Image2D
from cuqi.operator import FirstOrderFiniteDifference
from cuqi.core import Distribution

class Laplace_diff(Distribution):
    """Laplace distribution on the difference between neighboring nodes.

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
        prior = cuqi.distribution.Laplace_diff(location=np.zeros(128), scale=0.1)

    Notes
    -----
    The pdf is given by

    .. math::

        \pi(\mathbf{x}) = \\frac{1}{(2b)^n} \exp \left(- \\frac{\|\mathbf{D}(\mathbf{x}-\mathbf{x}_0) \|_1 }{b} \\right),

    where :math:`\mathbf{x}_0\in \mathbb{R}^n` is the location parameter, :math:`b` is the scale, :math:`\mathbf{D}` is the difference operator.
 
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

        elif physical_dim == 1:
            num_nodes = self.dim
        else:
            raise ValueError("Only physical dimension 1 or 2 supported.")

        self._diff_op = FirstOrderFiniteDifference(num_nodes=num_nodes, bc_type=bc_type)

    def pdf(self, x):
        Dx = self._diff_op @ (x-self.location)  # np.diff(X)
        return (1/(2*self.scale))**(len(Dx)) * np.exp(-np.linalg.norm(Dx, ord=1, axis=0)/self.scale)

    def logpdf(self, x):
        Dx = self._diff_op @ (x-self.location)
        return len(Dx)*(-(np.log(2)+np.log(self.scale))) - np.linalg.norm(Dx, ord=1, axis=0)/self.scale

    def _sample(self,N=1,rng=None):
        raise NotImplementedError("'Laplace_diff.sample' is not implemented. Sampling can be performed with the 'sampler' module.")

    # def cdf(self, x):   # TODO
    #     return 1/2 + 1/2*np.sign(x-self.loc)*(1-np.exp(-np.linalg.norm(x, ord=1, axis=0)/self.scale))

    # def sample(self):   # TODO
    #     p = np.random.rand(self.dim)
    #     return self.loc - self.scale*np.sign(p-1/2)*np.log(1-2*abs(p-1/2))