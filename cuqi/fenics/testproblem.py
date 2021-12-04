class FEniCSDiffusion(PDEModel):

    def __init__(self, mesh, Vh, bc=None, bc0=None, f= None,\
        measurement_type = 'potential', parameter_type="conductivity_field"):


        if parameter_type == "fixed_radius_inclusion": #TODO: Add function to create form based on parameter_type 
            R = 0.1
            Gamma = 1
            tol = 1E-6  
            self.kappa = lambda m: dl.Expression('pow(x[0]-m0,2)+pow(x[1]-m1,2) <= pow(R,2) + tol ? 1+Gamma : 1', \
                                  degree=1, tol=tol, m0=m.vector()[0], m1=m.vector()[1], R=R, Gamma = Gamma)#TODO: make the form differentiable with respect to m
            domain_geometry = cuqi.geometry.Discrete(['x','y'])
            def form(u,m,p):
                return self.kappa(m)*ufl.inner(ufl.grad(u), ufl.grad(p))*ufl.dx - f*p*ufl.dx 

        elif parameter_type == "conductivity_field":
            domain_geometry = cuqi.fenics.geometry.FenicsContinuous(Vh[1])
            def form(u,m,p):
                return ufl.exp(m)*ufl.inner(ufl.grad(u), ufl.grad(p))*ufl.dx - f*p*ufl.dx 

                #self, form, mesh, Vh, bc=None, bc0=None, obs_op=None
        obs_op = self._create_obs_op(measurement_type)
        super().__init__(form, mesh, Vh, bc=bc, bc0=bc0, f=f, obs_op = obs_op, 
                         range_geometry=None , domain_geometry = domain_geometry)
    

    def _create_obs_op(cls, measurement_type):

        if measurement_type == 'potential':
            obs_op = lambda m, u: u 
        elif measurement_type == 'gradu_squared':
            obs_op = lambda m, u: dl.inner(dl.grad(u),dl.grad(u))
        elif measurement_type == 'power_density':
            obs_op = lambda m, u: dl.exp(m)*dl.inner(dl.grad(u),dl.grad(u))
        elif measurement_type == 'sigma_u':
            obs_op = lambda m, u: dl.exp(m)*u
        elif measurement_type == 'sigma_norm_gradu':
            obs_op = lambda m, u: dl.exp(m)*dl.sqrt(dl.inner(dl.grad(u),dl.grad(u)))
        else:
            raise NotImplementedError
        return obs_op

    
class FEniCSDiffusion1D(FEniCSDiffusion):

    def __init__(self, mesh=None, Vh=None, bc=None, bc0=None, f=None,\
                 measurement_type = 'potential', parameter_type="conductivity_field", shape= (50,)):
        if shape is not None:
            if len(shape) !=1: 
                raise ValueError('shape need to be a one dimensional tuple (e.g. (20,))') 
            N = np.prod(shape)
        else: 
            N = 30

        if mesh is None: 
            mesh = dl.UnitIntervalMesh(N)
        elif (isinstance(mesh, tuple) and len(mesh) ==1) or isinstance(mesh, (int,np.integer)):
            if isinstance(mesh, tuple): mesh = mesh[0] 
            mesh = dl.UnitIntervalMesh(mesh)

        if Vh is None:
            Vh_STATE = dl.FunctionSpace(mesh, 'Lagrange', 1)
            Vh_PARAMETER = dl.FunctionSpace(mesh, 'Lagrange', 1)
            Vh_ADJOINT = dl.FunctionSpace(mesh, 'Lagrange', 1)
            Vh = [Vh_STATE, Vh_PARAMETER, Vh_ADJOINT]

        if bc is None:
            assert(bc0 == None), "If bc0 is specified, bc must be specified as well."
            def u_boundary(x, on_boundary):
                return on_boundary

            u_bdr = dl.Expression("x[0]", degree=1)
            bc = dl.DirichletBC(Vh_STATE, u_bdr, u_boundary)

            u_bdr0 = dl.Constant(0.0)
            bc0 = dl.DirichletBC(Vh_ADJOINT, u_bdr0, u_boundary)

        if f is None:
            f = dl.Constant(0.0)

        super().__init__(mesh, Vh, bc=bc, bc0=bc0, f=f, measurement_type = measurement_type,\
                         parameter_type=parameter_type)

class FEniCSDiffusion2D(FEniCSDiffusion):

    def __init__(self, mesh=None, Vh=None, bc=None, bc0=None, f=None,\
                 measurement_type = 'potential', parameter_type="conductivity_field"):
        if mesh is None:
            mesh = dl.UnitSquareMesh(40, 40)
        elif isinstance(mesh, tuple) and len(mesh) ==2: 
            mesh = dl.UnitSquareMesh(mesh[0], mesh[1])

        if Vh is None: #TODO: creating spaces should be part of the super class (diffusion)
            Vh_STATE = dl.FunctionSpace(mesh, 'Lagrange', 1)
            if parameter_type=="conductivity_field":
                Vh_PARAMETER = dl.FunctionSpace(mesh, 'Lagrange', 1)
            elif parameter_type == "fixed_radius_inclusion":    
                Vh_PARAMETER = dl.VectorFunctionSpace(mesh, "R", degree=0, dim=2)
            Vh_ADJOINT = dl.FunctionSpace(mesh, 'Lagrange', 1)
            Vh = [Vh_STATE, Vh_PARAMETER, Vh_ADJOINT]

        if bc is None:
            assert(bc0 == None), "If bc0 is specified, bc must be specified as well."
            def u_boundary(x, on_boundary):
                return on_boundary

            u_bdr = dl.Expression("x[0]", degree=1)
            bc = dl.DirichletBC(Vh_STATE, u_bdr, u_boundary)

            u_bdr0 = dl.Constant(0.0)
            bc0 = dl.DirichletBC(Vh_ADJOINT, u_bdr0, u_boundary)

        if f is None:
            f = dl.Constant(0.0)

        super().__init__(mesh, Vh, bc=bc, bc0=bc0, f=f, measurement_type = measurement_type,\
                         parameter_type=parameter_type)

