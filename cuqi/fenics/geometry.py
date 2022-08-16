from cuqi.geometry import Geometry, MappedGeometry, _WrappedGeometry
import numpy as np
import matplotlib.pyplot as plt
import dolfin as dl
import ufl

__all__ = [
    'FEniCSContinuous',
    'FEniCSMappedGeometry',
    'MaternExpansion'
]

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

    def fun2par(self,fun):
        """ Map the function values (FEniCS object) to the corresponding parameters (ndarray)."""
        return fun.vector().get_local()

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


class MaternExpansion(_WrappedGeometry):
    """A geometry class that builds spectral representation of Matern covariance operator on the given input geometry. We create the representation using the stochastic partial differential operator, equation (15) in (Roininen, Huttunen and Lasanen, 2014). Zero Neumann boundary conditions are assumed for the stochastic partial differential equation (SPDE) and the smoothness parameter :math:`\\nu` is set to 1. To generate Matern field realizations, the method :meth:`par2field` is used. The input `p` of this method need to be an `n=dim` i.i.d random variables that follow a normal distribution. 

    For more details about the formulation of the SPDE see: Roininen, L., Huttunen, J. M., & Lasanen, S. (2014). Whittle-Matérn priors for Bayesian statistical inversion with applications in electrical impedance tomography. Inverse Problems & Imaging, 8(2), 561.

    Parameters
    -----------
    geometry : cuqi.fenics.geometry.Geometry
        An input geometry on which the Matern field representation is built (the geometry must have a mesh attribute)

    length_scale : float
        Length scale paramater (controls correlation length)

    num_terms: int
        Number of expantion terms to represent the Matern field realization


    Example
    -------
    .. code-block:: python

        import numpy as np
        import matplotlib.pyplot as plt
        from cuqi.fenics.geometry import MaternExpansion, FEniCSContinuous
        from cuqi.distribution import GaussianCov
        import dolfin as dl
        
        mesh = dl.UnitSquareMesh(20,20)
        V = dl.FunctionSpace(mesh, 'CG', 1)
        geometry = FEniCSContinuous(V)
        MaternGeometry = MaternExpansion(geometry, 
                                        length_scale = .2,
                                        num_terms=128)
        
        MaternField = GaussianCov(np.zeros(MaternGeometry.dim),
                        cov=np.eye(MaternGeometry.dim),
                        geometry= MaternGeometry)
        
        samples = MaternField.sample()
        samples.plot()

    """

    def __init__(self, geometry, length_scale, num_terms): 
        super().__init__(geometry)
        if not hasattr(geometry, 'mesh'):
            raise NotImplementedError
        self._length_scale = length_scale
        self._nu = 1
        self._num_terms = num_terms
        self._eig_val = None
        self._eig_vec = None

    @property
    def shape(self):
        return (self.num_terms,)

    @property
    def length_scale(self):
        return self._length_scale

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

    def __repr__(self) -> str:
        return "{} on {}".format(self.__class__.__name__,self.geometry.__repr__())

    def par2fun(self,p):
        return self.geometry.par2fun(self.par2field(p))

    def par2field(self, p):
        """Applies linear transformation of the parameters p to
        generate a realization of the Matern field (given that p is a
        sample of `n=dim` i.i.d random variables that follow a normal
        distribution)"""

        if self._eig_vec is None and self._eig_val is None:
            self._build_basis() 
	   
        p = self._process_values(p)
        Ns = p.shape[-1]
        field_list = np.empty((self.geometry.dim,Ns))

        for idx in range(Ns):
            field_list[:,idx] = self.eig_vec@( np.sqrt(self.eig_val)*p[...,idx] )

        if len(field_list) == 1:
            return field_list[0]
        else:
            return field_list

    def _build_basis(self):
        """Builds the basis of expansion of the Matern covariance operator"""
        V = self._build_space()
        u = dl.TrialFunction(V)
        v = dl.TestFunction(V)

        tau2 = 1/self.length_scale/self.length_scale
        a = tau2*u*v*dl.dx + dl.inner(dl.grad(u), dl.grad(v))*dl.dx

        A = dl.assemble(a)
        mat = A.array()

        eig_val, eig_vec = np.linalg.eig(mat)
        eig_val = np.reciprocal(np.real(eig_val))
        eig_vec = np.real(eig_vec)
        indices = np.argsort(eig_val)[::-1]
        eig_val = eig_val[indices]
        eig_vec = eig_vec[:,indices]
        self._eig_val = eig_val[:self.num_terms]
        self._eig_vec = eig_vec[:,:self.num_terms]

    def _build_space(self):
        """Create the function space on which the Matern covariance is discretized"""

        if hasattr(self.geometry, 'mesh'): 
            mesh = self.geometry.mesh
            V = dl.FunctionSpace(mesh, "CG", 1)
	
        else:
            raise NotImplementedError

        return V