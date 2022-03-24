
import numpy as np
import scipy.linalg as linalg
import dolfin as dl
from mshr import *
from cuqi.geometry import MappedGeometry
from cuqi.fenics.geometry import FEniCSMappedGeometry


class Field(MappedGeometry):
    def __init__(self, geometry):  # points can be mesh or grid
        super().__init__(geometry, map=self.par2field)

    def par2field(self, p):
        raise NotImplementedError

    def par2fun(self,p):
        return self.geometry.par2fun(self.map(p))

    def fun2par(self,f):
        raise NotImplementedError


class Matern(Field):

    def __init__(self, geometry, l, nu, num_terms):  # points can be mesh or grid
        super().__init__(geometry)
        self._l = l
        self._nu = nu
        self._num_terms = num_terms
        self._eig_val = None
        self._eig_vec = None

    @property
    def dim(self):
        return self.num_terms

    @property
    def l(self):
        return self._l

    @property
    def nu(self):
        return self._nu

    @property
    def num_terms(self):
        return self._num_terms

    @property
    def eig_val(self):
        return self._eig_val

    @property
    def eig_vec(self):
        return self._eig_vec

    def __call__(self, p):
        return self.par2field(p)

    def _process_values(self,values):
        if len(values.shape) == 3 or\
             (len(values.shape) == 2 and values.shape[0]== self.dim):  
            pass
        else:
            values = values[..., np.newaxis]
        return values


    def par2field(self, p):
        # is it mesh or grid
        # construct basis and keep them in memory
        # robust way to know when basis need to be updated
        if self._eig_vec is None and self._eig_val is None:
            self._build_basis() 
	   
        p = self._process_values(p)
        Ns = p.shape[-1]
        field_list = np.empty((self.geometry.dim,Ns))

        for idx in range(Ns):
            field_list[:,idx] = self.eig_vec@( self.eig_val*p[...,idx] )

        if len(field_list) == 1:
            return field_list[0]
        else:
            return field_list


    def _build_basis(self):
        V = self._build_space()
        u = dl.TrialFunction(V)
        v = dl.TestFunction(V)

        tau2 = 1/self.l/self.l
        a = tau2*u*v*dl.dx - dl.inner(dl.grad(u), dl.grad(v))*dl.dx

        A = dl.assemble(a)
        mat = A.array()

        print('creating kernel ...')
        self.Ker = np.linalg.matrix_power(mat, -self.nu)

        print('computing eigen decomposition ...')
        eig_val, eig_vec = np.linalg.eig(self.Ker)
        eig_val = np.real(eig_val)
        eig_vec = np.real(eig_vec)
        self._eig_val = eig_val[:self.num_terms]
        self._eig_vec = eig_vec[:,:self.num_terms]

    def _build_space(self):

        if hasattr(self.geometry, 'mesh'): 
            mesh = self.geometry.mesh
            V = dl.FunctionSpace(mesh, "CG", 1)
	
        else:
            raise NotImplementedError

        #elif hasattr(self.geometry, 'grid') :
	#    X,Y = np.meshgrid(self.geometry.grid[0], self.geometry.grid[1])
	#    self.grid_cooor = np.vstack(( Y.flatten(), X.flatten())).T
	#    mesh = 
        return V