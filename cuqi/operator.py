import numpy as np
from scipy.sparse import diags, spdiags, eye, kron, vstack


# ========== Covariance class ===========
class Operator(object):

    def __init__(self):
        pass

class FirstOrderFiniteDifference(Operator):
    def __init__(self, N  , bc_type= 'periodic', dom = 1):	
        # finite difference matrix
        self.dom = dom
        self.N = N
        self.bc_type = bc_type
        one_vec = np.ones(N)
        diags = np.vstack([-one_vec, one_vec])
        if (bc_type == 'zero'):
            locs = [-1, 0]
            Dmat = spdiags(diags, locs, N+1, N)
        elif (bc_type == 'periodic'):
            locs = [-1, 0]
            Dmat = spdiags(diags, locs, N+1, N).tocsr()
            Dmat[-1, 0] = 1
            Dmat[0, -1] = -1
        elif (bc_type == 'neumann'):
            locs = [0, 1]
            Dmat = spdiags(diags, locs, N-1, N)
        elif (bc_type == 'backward'):
            locs = [0, -1]
            Dmat = spdiags(diags, locs, N, N).tocsr()
            Dmat[0, 0] = 1
        elif (bc_type == 'none'):
            Dmat = eye(N)

                # structure matrix
        if (dom == 1):
            self.dim = N
            self.D = Dmat

        elif (dom == 2):            
            self.dim = N**2
            I = eye(N, dtype=int)
            self.Ds = kron(I, Dmat)
            self.Dt = kron(Dmat, I)
            self.D = vstack([self.Ds, self.Dt])


class PrecisionFiniteDifference(Operator):
    def __init__(self, N  , bc_type= 'periodic', dom = 1, order =1):
        self.dom = dom
        self.N = N
        self.bc_type = bc_type	
        if order == 1:
            self._FOFD = FirstOrderFiniteDifference(N, bc_type=bc_type, dom = dom)
        else:
            raise NotImplementedError


        if (dom == 1):
            self.L = (self.D.T @ self.D).tocsc()
        elif (dom == 2):            
            self.L = ((self.Ds.T @ self.Ds) + (self.Dt.T @ self.Dt)).tocsc()

    @property
    def Dt(self):
        if self.dom == 2:
            return self._FOFD.Dt
        else:
            raise AttributeError("Dt is only defined for the case in which dom=2.") 

    @property
    def Ds(self):
        if self.dom == 2:
            return self._FOFD.Ds
        else:
            raise AttributeError("Ds is only defined for the case in which dom=2.")  

    @property
    def D(self):
        return self._FOFD.D

    @property
    def dim(self):
        return self._FOFD.dim


