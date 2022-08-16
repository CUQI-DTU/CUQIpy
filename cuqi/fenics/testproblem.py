import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("../../")
import dolfin as dl
import warnings
import cuqi
from cuqi.problem import BayesianProblem
import dolfin as dl
import ufl

__all__ = ['FEniCSDiffusion1D']

class FEniCSDiffusion1D(BayesianProblem):
    """
    1D Diffusion PDE problem using FEniCS. The problem has Dirichlet boundary conditions.

    Parameters
    ------------
    dim : int, Default 100
        Dimension of the 1D problem

    endpoint : float, Default 1
        Endpoint of the 1D grid (starts at 0).

    exactSolution : ndarray, CUQIarray, Default None
        Exact solution used to generate data. 
        If None a default exact solution is chosen.

    SNR : float, Default 100
        Signal-to-noise ratio.
    
    mapping: str or function
        mapping to parametrize the Bayesain parameters. If provided as string, it can take one of the values: ['exponential'] 

    Attributes
    ----------
    data : ndarray
        Generated (noisy) data

    model : cuqi.model.Model
        Deconvolution forward model

    prior : cuqi.core.Distribution
        Distribution of the prior (Default = None)

    likelihood : cuqi.likelihood.Likelihood
        Likelihood function.

    exactSolution : ndarray
        Exact solution (ground truth)

    exactData : ndarray
        Noise free data

    infoSring : str
        String with information about the problem, noise etc.

    Methods
    ----------
    MAP()
        Compute MAP estimate of posterior.
        NB: Requires prior to be defined.

    sample_posterior(Ns)
        Sample Ns samples of the posterior.
        NB: Requires prior to be defined.

    """
    
    def __init__(self, dim = 100, endpoint = 1, exactSolution = None, SNR = 100, observation_operator = None, mapping = None, left_bc=0, right_bc=1):

        # Create FEniCSPDE        
        def u_boundary(x, on_boundary):
            return on_boundary

        def form(m,u,p):
            return m*ufl.inner(ufl.grad(u), ufl.grad(p))*ufl.dx
        
        mesh = dl.IntervalMesh(dim, 0,endpoint)

        solution_function_space = dl.FunctionSpace(mesh, 'Lagrange', 1)
        parameter_function_space = dl.FunctionSpace(mesh, 'Lagrange', 1)

        dirichlet_bc_expression = dl.Expression("left_bc*(x[0]<eps)+right_bc*(x[0]>endpoint-eps)", eps=dl.DOLFIN_EPS, endpoint=endpoint, left_bc=left_bc, right_bc=right_bc, degree=1)
        dirichlet_bc = dl.DirichletBC(solution_function_space, dirichlet_bc_expression, u_boundary)

        PDE = cuqi.fenics.pde.SteadyStateLinearFEniCSPDE( form, mesh, solution_function_space, parameter_function_space, dirichlet_bc,observation_operator=observation_operator)
        
        # Create PDE model
        domain_geometry = cuqi.fenics.geometry.FEniCSContinuous(parameter_function_space)
        if mapping is not None:
            if mapping == 'exponential':
                mapping = lambda x : ufl.exp(x)
            elif callable(mapping):
                mapping = mapping
            else:
                raise ValueError('mapping can be a function or one of the expected keywords.')
            domain_geometry =  cuqi.fenics.geometry.FEniCSMappedGeometry(geometry=domain_geometry, map = mapping)

        range_geometry = cuqi.fenics.geometry.FEniCSContinuous(solution_function_space)
        
        model = cuqi.model.PDEModel(PDE,range_geometry,domain_geometry)

        # Create prior
        pr_mean = np.zeros(domain_geometry.dim)
        prior = cuqi.distribution.GMRF(pr_mean,25,1,'zero') 
        
        # Set up exact solution
        if exactSolution is None:
            exactSolution = prior.sample()
        elif exactSolution == 'smooth_step':
            N = dim + 1
            fun = lambda grid:  0.8*np.exp( -( (grid -endpoint/2.0 )**2 ) / 0.02)
            grid = np.linspace(0,endpoint,N)
            exactSolution = np.ones(N)*.8
            exactSolution[np.where(grid>endpoint/2.0)] = fun(grid[np.where(grid>endpoint/2.0)]) 

        # Generate exact data
        b_exact = model.forward(domain_geometry.par2fun(exactSolution),is_par=False)

        # Add noise to data
        sigma = np.linalg.norm(b_exact)/SNR
        sigma2 = sigma*sigma # variance of the observation Gaussian noise
        data = b_exact + np.random.normal( 0, sigma, b_exact.shape )
        
        # Create likelihood
        likelihood = cuqi.distribution.GaussianCov(model, sigma2*np.eye(range_geometry.dim)).to_likelihood(data)

        # Initialize FEniCSDiffusion1D as BayesianProblem problem
        super().__init__(likelihood, prior)

        # Store exact values and information
        self.exactSolution = exactSolution
        self.exactData = b_exact
        self.infoString = f"Noise type: Additive i.i.d. noise with mean zero and signal to noise ratio: {SNR}"

