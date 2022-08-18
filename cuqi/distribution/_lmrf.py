from cuqi.distribution import Distribution
from cuqi.operator import FirstOrderFiniteDifference
import numpy as np

class LMRF(Distribution):
    """
        Parameters
        ----------
        partition_size : int
            The dimension of the distribution in one physical dimension. 

        physical_dim : int
            The physical dimension of what the distribution represents (can take the values 1 or 2).
    """
        
    def __init__(self, mean, prec, partition_size, physical_dim, bc_type, **kwargs):
        super().__init__(**kwargs)
        self.mean = mean.reshape(len(mean), 1)
        self.prec = prec
        self._partition_size = partition_size          # partition size
        self._bc_type = bc_type      # boundary conditions
        self._physical_dim = physical_dim
        if physical_dim == 1: 
            num_nodes = (partition_size,) 
        else:
            num_nodes = (partition_size,partition_size)

        self._diff_op = FirstOrderFiniteDifference( num_nodes, bc_type= bc_type) 

    @property
    def dim(self):
        return self._diff_op.dim

    def logpdf(self, x):

        if self._physical_dim == 1 or self._physical_dim == 2:
            const = self.dim *(np.log(self.prec)-np.log(2)) 
            y = const -  self.prec*(np.linalg.norm(self._diff_op@x, ord=1))
        else:
            raise NotImplementedError
        return y

    def _sample(self, N):
        raise NotImplementedError