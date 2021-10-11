import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse import hstack

import cuqi
from cuqi.samples import Samples
from cuqi.model import Model
import warnings

try: 
    import dolfin as dl
    import ufl
except Exception as error:
    warnings.warn(error.msg)

class FEniCSPDEModel(cuqi.model.Model):
    """
    Parameters
    ----------
    forward : Problem variational form (includes boundary conditions and source terms)
    mesh : FEniCS Mesh
    Vh : A python list of Function spaces of the state variables, parameters, and adjoint variables
    bc : forward problem boundary conditions (Dirichlet)
    bc0: adjoint problem boundary conditions (Dirichlet)
    """
    def __init__(self, form, mesh, Vh, bc=None, bc0=None, f=None, obs_op=None, range_geometry=None, domain_geometry=None):
        if range_geometry is None: range_geometry = cuqi.fenicsGeometry.FenicsContinuous(Vh[0])
        super().__init__(self._forward_func,range_geometry=range_geometry,domain_geometry=domain_geometry)
        self.form = form
        self.mesh = mesh
        self.Vh  = Vh
        self.bc  = bc
        self.bc0 = bc0
        self.dim = Vh[0].dim()
        self.obs_op = obs_op 
        self.spatial_dim = mesh.topology().dim()

    def _solve(self, m):
        """
        Input:
        ----------
        m: Bayesian problem parameter

        Output:
        ----------
        u: solution

        """
        warnings.warn("_solve is implemented for steady_state linear PDE. "+\
                       "The linearity is with respect to the state variables.")

        m_fun = dl.Function(self.Vh[1])
        m_fun.vector().set_local(m) 
        Vh = self.Vh
        u = dl.TrialFunction(Vh[0])
        p = dl.TestFunction(Vh[0])
        a, L  = dl.lhs(self.form(u,m_fun,p)), dl.rhs(self.form(u,m_fun,p))
        u_sol = dl.Function(self.Vh[0])
        dl.solve(a == L, u_sol, self.bc)
        return u_sol


    def _forward_func(self, m):
        """
        Input:
        ----------
        m: Bayesian problem parameter

        Output:
        ----------
        b: observables, the result of applying the observation operator to the PDE solution

        """
        u_sol = self._solve(m)
        if self.obs_op == None: 
            return u_sol.vector().get_local()
        else:
            return self.apply_obs_op([u_sol.vector(),m])

    def apply_obs_op(self, x):
        m_fun = dl.Function(self.Vh[1])
        m_fun.vector().set_local(x[1])
        u_fun = dl.Function(self.Vh[0])
        u_fun.vector().set_local(x[0]) 

        obs = self.obs_op(m_fun, u_fun)
        if isinstance(obs, ufl.algebra.Operator):
            return dl.project(obs, self.Vh[0]).vector().get_local()
        elif isinstance(obs, dl.function.function.Function):
            return obs.vector().get_local()
        elif isinstance(obs, (np.ndarray, int, float)):
            return obs
        else:
            raise NotImplementedError("obs_op output must be a number, a numpy array or a ufl.algebra.Operator type")

    def eval_param(self, DOF, X, Y):
        m_fun = dl.Function(self.Vh[1])
        m_fun.vector().set_local(DOF)
        Z = np.empty_like(X).flatten()
        for idx, (x,y) in enumerate(zip(X.flatten(),Y.flatten())):
            #print(x,y)
            Z[idx] = m_fun(x,y)
        Z = Z.reshape(X.shape)
        return Z

    def eval_data(self, DOF, X, Y):
        u_fun = dl.Function(self.Vh[0])
        u_fun.vector().set_local(DOF)
        Z = np.empty_like(X).flatten()
        for idx, (x,y) in enumerate(zip(X.flatten(),Y.flatten())):
            #print(x,y)
            Z[idx] = u_fun(x,y)
        Z = Z.reshape(X.shape)
        return Z




class FEniCSDiffusion(FEniCSPDEModel):

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
            domain_geometry = cuqi.fenicsGeometry.FenicsContinuous(Vh[1])
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
                 measurement_type = 'potential', parameter_type="conductivity_field"):
        if mesh is None:
            mesh = dl.UnitIntervalMesh(50) 

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

    def grid4param_plot(self, DOF=None, x_res = 400, y_res = 400): #TODO: Make 2D diffusion inherit from FEniCSDiffusion & 
                                         # FEniCS2D. grid4param_plot should be in FEniCS2D 
        #TODO: x_res and y_res should be mesh dependant and can be passed by user

        x_min = np.min(self.mesh.coordinates()[:,0]) 
        x_max = np.max(self.mesh.coordinates()[:,0])
        y_min = np.min(self.mesh.coordinates()[:,1]) 
        y_max = np.max(self.mesh.coordinates()[:,1])
        X, Y =  np.meshgrid(np.linspace(x_min, x_max,x_res), \
                           np.linspace(y_min, y_max,y_res))
        if DOF is None:
            return X,Y
        else:
            Z = self.eval_param(DOF, X, Y)
            return X,Y,Z

    def grid4data_plot(self, DOF=None, x_res = 400, y_res = 400): #TODO: Make 2D diffusion inherit from FEniCSDiffusion & 
                                         # FEniCS2D. grid4param_plot should be in FEniCS2D 
        #TODO: x_res and y_res should be mesh dependant and can be passed by user
        x_min = np.min(self.mesh.coordinates()[:,0]) 
        x_max = np.max(self.mesh.coordinates()[:,0])
        y_min = np.min(self.mesh.coordinates()[:,1]) 
        y_max = np.max(self.mesh.coordinates()[:,1])
        X, Y =  np.meshgrid(np.linspace(x_min, x_max,x_res), \
                           np.linspace(y_min, y_max,y_res))
        if DOF is None:
            return X,Y
        else:
            Z = self.eval_data(DOF, X, Y)
            return X,Y,Z

