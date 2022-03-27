from cuqi.geometry import Discrete, Geometry, MappedGeometry, _GeometryWrapper
import numpy as np
import matplotlib.pyplot as plt
import dolfin as dl
import ufl

class FEniCSContinuous(Geometry):

    def __init__(self, function_space, labels = ['x', 'y']):
        self.function_space = function_space
        if self.physical_dim >2:
            raise NotImplementedError("'FenicsContinuous' object does not support 3D meshes yet. 'mesh' needs to be a 1D or 2D mesh.")
        self.labels = labels

    @property
    def physical_dim(self):
        return self.function_space.mesh().geometry().dim()  

    @property
    def mesh(self):
        return self.function_space.mesh()

    @property
    def shape(self):
        return (self.function_space.dim(),)

    def par2fun(self,par):
        """The parameter to function map used to map parameters to function values in e.g. plotting."""
        par = self._process_values(par)
        Ns = par.shape[-1]
        fun_list = []
        for idx in range(Ns):
            fun = dl.Function(self.function_space)
            fun.vector().zero()
            fun.vector().set_local(par[...,idx])
            fun_list.append(fun)

        if len(fun_list) == 1:
            return fun_list[0]
        else:
            return fun_list

    def _plot(self,values,subplots=True,**kwargs):
        """
        Overrides :meth:`cuqi.geometry.Geometry.plot`. See :meth:`cuqi.geometry.Geometry.plot` for description  and definition of the parameter `values`.
        
        Parameters
        -----------
        kwargs : keyword arguments
            keyword arguments which the function :meth:`dolfin.plot` normally takes.
        """
        if isinstance(values, dl.function.function.Function):
            Ns = 1
            values = [values]
        elif hasattr(values,'__len__'): 
            Ns = len(values)
        subplot_ids = self._create_subplot_list(Ns,subplots=subplots)

        ims = []
        for rows,cols,subplot_id in subplot_ids:
            fun = values[subplot_id-1]
            if subplots:
                plt.subplot(rows,cols,subplot_id); 
            ims.append(dl.plot(fun, **kwargs))

        self._plot_config(subplots) 
        return ims

    def _process_values(self, values):
        if isinstance(values, dl.function.function.Function):
            return [values]

        elif len(values.shape) == 1:
            values = values[..., np.newaxis]
        
        return values
    
    def _plot_config(self, subplot):
        if self.labels is not None:
            if subplot == False:
                plt.gca().set_xlabel(self.labels[0])
                if self.physical_dim == 2: plt.gca().set_ylabel(self.labels[1]) 
            else:
                for i, axis in enumerate(plt.gcf().axes):
                    axis.set_xlabel(self.labels[0])
                    if self.physical_dim == 2: axis.set_ylabel(self.labels[1])


class FEniCSMappedGeometry(MappedGeometry):
    """
    """
    def par2fun(self,p):
        funvals = self.geometry.par2fun(p)
        if isinstance(funvals, dl.function.function.Function):
            funvals = [funvals]
        mapped_value_list = []
        for idx in range(len(funvals)):
            mapped_value = self.map(funvals[idx]) 
            if isinstance(mapped_value, ufl.algebra.Operator):
                mapped_value_list.append(dl.project(mapped_value, self.geometry.function_space))
            elif isinstance(mapped_value,dl.function.function.Function):
                mapped_value_list.append(mapped_value)
            else:
                raise ValueError(f"'{self.__class__.__name__}.map' should return 'ufl.algebra.Operator'")
            
        if len(mapped_value_list) == 1:
            return mapped_value_list[0]
        else:
            return mapped_value_list
    
    def fun2par(self,f):
        raise NotImplementedError


class Matern(_GeometryWrapper):

    def __init__(self, geometry, l, nu, num_terms):  # points can be mesh or grid
        super().__init__(geometry)
        if not hasattr(geometry, 'mesh'):
            raise NotImplementedError
        self._l = l
        self._nu = nu
        self._num_terms = num_terms
        self._eig_val = None
        self._eig_vec = None

    @property
    def shape(self):
        return (self.num_terms,)

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

    def par2fun(self,p):
        return self.geometry.par2fun(self.par2field(p))

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