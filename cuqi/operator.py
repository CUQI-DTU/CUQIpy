import numpy as np
from scipy.sparse import spdiags, eye, kron, vstack


# ========== Covariance class ===========
class Operator(object):

    def __init__(self):
        self._matr = None
        pass

    def __matmul__(self, vec):
        return self._matr@vec

    def __rmatmul__(self, vec):
        return vec@self._matr

    def __add__(self, val):
        return self._matr+val

    def __radd__(self, val):
        return self.__add__(val)

    def __mul__(self, val):
        return self._matr*val

    def __rmul__(self, val):
        return self.__mul__(val)

    @property
    def T(self):
        return self._matr.T

    @property
    def shape(self):
        return self._matr.shape

class FirstOrderFiniteDifference(Operator):

    def __init__(self, num_nodes, bc_type= 'periodic'):	

        if isinstance(num_nodes, (int,np.integer)):
            self._num_nodes = (num_nodes,)

        elif isinstance(num_nodes, tuple) and len(num_nodes) in [1,2] and\
            np.all([isinstance(num,(int,np.integer)) for num in num_nodes]):
            
            if len(num_nodes)==2 and (num_nodes[0] != num_nodes[1]):
                raise NotImplementedError("The case in which the number of nodes in the x direction is not equal to the number of nodes in the y direction is not implemented.")
            self._num_nodes = num_nodes
        
        else:
            raise ValueError("num_nodes should be either and integer or a two dimensional tuple of integers")

        self._bc_type = bc_type
        self._create_diff_matrix()


    @property
    def num_nodes(self):
        return self._num_nodes

    @property
    def physical_dim(self):
        return len(self.num_nodes)

    @property
    def bc_type(self):
        return self._bc_type

    @property
    def dim(self):
        return np.prod(self.num_nodes)

    @property
    def _matr(self):
        return self._D


    def _create_diff_matrix(self):
        if self.physical_dim == 2:
            assert(self.num_nodes[0] == self.num_nodes[1]), "The case in which self.num_nodes[0] != self.num_nodes[1] is not handled."
            N = self.num_nodes[0]
        elif self.physical_dim == 1:
            N = self.num_nodes[0]
        else:
            raise Exception("Cannot define N")

        # finite difference matrix
        one_vec = np.ones(N)
        diags = np.vstack([-one_vec, one_vec])
        if (self.bc_type == 'zero'):
            locs = [-1, 0]
            Dmat = spdiags(diags, locs, N+1, N)
        elif (self.bc_type == 'periodic'):
            locs = [-1, 0]
            Dmat = spdiags(diags, locs, N+1, N).tocsr()
            Dmat[-1, 0] = 1
            Dmat[0, -1] = -1
        elif (self.bc_type == 'neumann'):
            locs = [0, 1]
            Dmat = spdiags(diags, locs, N-1, N)
        elif (self.bc_type == 'backward'):
            locs = [0, -1]
            Dmat = spdiags(diags, locs, N, N).tocsr()
            Dmat[0, 0] = 1
        elif (self.bc_type == 'none'):
            Dmat = eye(N)

        # structure matrix
        if (self.physical_dim == 1):
            self._D = Dmat

        elif (self.physical_dim == 2):            
            I = eye(N, dtype=int)
            self._Ds = kron(I, Dmat)
            self._Dt = kron(Dmat, I)
            self._D = vstack([self._Ds, self._Dt])
        


class PrecisionFiniteDifference(Operator):
    def __init__(self, num_nodes , bc_type= 'periodic', order =1):
        if order == 1:
            self._diff_op = FirstOrderFiniteDifference(num_nodes, bc_type=bc_type)
        else:
            raise NotImplementedError
        self._create_prec_matrix()

    @property
    def physical_dim(self):
        return self._diff_op.physical_dim

    @property
    def num_nodes(self):
        return self._diff_op.num_nodes

    @property
    def bc_type(self):
        return self._diff_op.bc_type

    @property
    def dim(self):
        return self._diff_op.dim

    @property
    def _matr(self):
        return self._L

    def _create_prec_matrix(self):
        if self.physical_dim == 1:
            self._L = (self._diff_op.T @ self._diff_op).tocsc()
        elif self.physical_dim == 2:            
            self._L = ((self._diff_op._Ds.T @ self._diff_op._Ds) + (self._diff_op._Dt.T @ self._diff_op._Dt)).tocsc()
        else:
            raise NotImplementedError


