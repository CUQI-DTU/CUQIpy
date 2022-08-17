import numpy as np
from scipy.sparse import spdiags, eye, kron, vstack

# ========== Operator class ===========
class Operator(object):
    """
    A linear operator which is represented by a matrix. 
    """

    def __init__(self):
        self._matrix = None
        pass

    def __matmul__(self, vec):
        return self._matrix@vec

    def __rmatmul__(self, vec):
        return vec@self._matrix

    def __add__(self, val):
        return self._matrix+val

    def __radd__(self, val):
        return self.__add__(val)

    def __mul__(self, val):
        return self._matrix*val

    def __rmul__(self, val):
        return self.__mul__(val)

    @property
    def T(self):
        return self._matrix.T

    @property
    def shape(self):
        return self._matrix.shape

    def get_matrix(self):
        return self._matrix

class FirstOrderFiniteDifference(Operator):
    """
    First order finite difference differential operator for 1D and 2D grids. 

    Attributes:
    -----------
        num_nodes: int or tuple
            For a 1D operator, num_nodes is a one dimensional tuple or an integer representing the number of discretization nodes in a 1D grid. For a 2D operator, num_nodes is a two dimensional tuple of integers representing the number of discretization nodes in the 2D grid in the x axis and the y axis, respectively.

        bc_type: str
            The boundary condition type for the operator. 

        physical_dim: int
            Either 1 or 2 for 1D and 2D operators, respectively.

        dx : int or float
            The grid spacing (length between two consecutive nodes). 
    """

    def __init__(self, num_nodes, bc_type= 'periodic', dx=None):	

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
        if dx == None:
            self._dx = 1
        elif self.physical_dim == 1:
            self._dx = dx #TODO: check dx is a scalar value
        else:
            raise NotImplementedError('Specifying dx for a 2D operator is not implemented')
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
        else:
            raise ValueError(f"Unknown boundary type {self.bc_type}")

        # structure matrix
        if (self.physical_dim == 1):
            self._matrix = Dmat/self._dx

        elif (self.physical_dim == 2):            
            I = eye(N, dtype=int)
            Ds = kron(I, Dmat)
            Dt = kron(Dmat, I)
            self._matrix = vstack([Ds, Dt])
        
class SecondOrderFiniteDifference(FirstOrderFiniteDifference):
    """
    Second order finite difference differential operator for 1D and 2D grids. 

    Attributes:
    -----------
        num_nodes: int or tuple
            For a 1D operator, num_nodes is a one dimensional tuple or an integer representing
            the number of discretization nodes in a 1D grid. For a 2D operator, num_nodes is a
            two dimensional tuple of integers representing the number of discretization nodes
            in the 2D grid in the x axis and the y axis, respectively.

        bc_type: str
            The boundary condition type for the operator. 

        physical_dim: int
            Either 1 or 2 for 1D and 2D operators, respectively.

        dx : int or float
            The grid spacing (length between two consecutive nodes). 
    """

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
        diags = np.vstack([-one_vec, 2*one_vec, -one_vec])
        if (self.bc_type == 'zero'):
            locs = [-2, -1, 0]
            Dmat = spdiags(diags, locs, N+2, N)
        elif (self.bc_type == 'periodic'):
            locs = [-2, -1, 0]
            Dmat = spdiags(diags, locs, N+2, N).tocsr()
            Dmat[0, -2] = -1
            Dmat[0:2, -1] = [2, -1]
            Dmat[-2, 0] = -1
            Dmat[-1, 0:2] = [2, -1]
        elif (self.bc_type == 'neumann'):
            locs = [0, 1, 2]
            Dmat = spdiags(diags, locs, N-2, N).tocsr()
        else:
            raise ValueError(f"Unknown boundary type {self.bc_type}")

        # structure matrix
        if (self.physical_dim == 1):
            self._matrix = Dmat/self._dx**2

        elif (self.physical_dim == 2):            
            I = eye(N, dtype=int)
            Ds = kron(I, Dmat)
            Dt = kron(Dmat, I)
            self._matrix = vstack([Ds, Dt])


class PrecisionFiniteDifference(Operator):
    """
    Precision operator constructed with a finite difference operator. 

    Attributes:
    -----------
        num_nodes: int or tuple
            For a 1D operator, num_nodes is a one dimensional tuple or an integer representing the number of discretization nodes in a 1D grid. For a 2D operator, num_nodes is a two dimensional tuple of integers representing the number of discretization nodes in the 2D grid in the x axis and the y axis, respectively.

        bc_type: str
            The boundary condition type for the finite difference operator. 

        physical_dim: int
            Either 1 or 2 for 1D and 2D operators, respectively. 
        
        order: int
            | The order of the finite difference operator.
            | Order 0: Identity operator.
            | Order 1: First order finite difference operator. 1D precision has a banded diagonal structure with [-1, 2, -1].
            | Order 2: Second order finite difference operator. 1D precision has a banded diagonal structure with [1, -4,  6, -4, 1].
    """
    def __init__(self, num_nodes , bc_type= 'periodic', order =1):
        if order == 0:
            self._diff_op = FirstOrderFiniteDifference(num_nodes, "none") # Special case that is idendity operator
        elif order == 1:
            self._diff_op = FirstOrderFiniteDifference(num_nodes, bc_type=bc_type)
        elif order == 2:
            self._diff_op = SecondOrderFiniteDifference(num_nodes, bc_type=bc_type)
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

    def _create_prec_matrix(self):
        if self.physical_dim == 1 or self.physical_dim == 2:
            self._matrix = (self._diff_op.T @ self._diff_op).tocsc()
        else:
            raise NotImplementedError


