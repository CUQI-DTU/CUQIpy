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


class FEniCSDiffusion1D(BayesianProblem):
    """
    1D Diffusion PDE problem using FEniCS.

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

    Attributes
    ----------
    data : ndarray
        Generated (noisy) data

    model : cuqi.model.Model
        Deconvolution forward model

    prior : cuqi.distribution.Distribution
        Distribution of the prior (Default = None)

    likelihood : cuqi.distribution.Distribution
        Distribution of the likelihood

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
    
    def __init__(self, dim = 100, L = 1, exactSolution = None, SNR = 100, observation_operator = None):

        def u_boundary(x, on_boundary):
            return on_boundary
        
        def form(m,u,p):
            return ufl.exp(m)*ufl.inner(ufl.grad(u), ufl.grad(p))*ufl.dx 
        
        #%% Create PDE model
        mesh = dl.IntervalMesh(dim, 0,L)
        solution_function_space = dl.FunctionSpace(mesh, 'Lagrange', 1)
        parameter_function_space = dl.FunctionSpace(mesh, 'Lagrange', 1)
        dirichlet_bc_expression = dl.Expression("x[0]", degree=1)
        dirichlet_bc = dl.DirichletBC(solution_function_space, dirichlet_bc_expression, u_boundary)
        PDE = cuqi.fenics.pde.SteadyStateLinearFEniCSPDE( form, mesh, solution_function_space, parameter_function_space, dirichlet_bc,observation_operator=observation_operator)

        domain_geometry = cuqi.fenics.geometry.FenicsContinuous(parameter_function_space)
        range_geometry = cuqi.fenics.geometry.FenicsContinuous(solution_function_space)
        
        # Prepare model
        model = cuqi.model.PDEModel(PDE,range_geometry,domain_geometry)

        pr_mean = np.zeros(domain_geometry.dim)
        prior = cuqi.distribution.GMRF(pr_mean,25,domain_geometry.dim,1,'zero') 
        # Set up exact solution
        if exactSolution is None:

            exactSolution = prior.sample()
            #np.sin(np.linspace(0,L,parameter_function_space.dim()))+ 2 

        # Generate exact data
        b_exact = model.forward(exactSolution,is_par=False)

        # Add noise to data
        sigma = np.linalg.norm(b_exact)/SNR
        sigma2 = sigma*sigma # variance of the observation Gaussian noise
        data = b_exact + np.random.normal( 0, sigma, b_exact.shape )

        likelihood = cuqi.distribution.GaussianCov(model, sigma2*np.eye(range_geometry.dim))

        
        #cuqi.distribution.GaussianCov(np.zeros(domain_geometry.dim), 1)

        # Initialize Deconvolution as BayesianProblem problem
        super().__init__(likelihood,prior,data)

        # Store exact values
        self.exactSolution = exactSolution
        self.exactData = b_exact
        self.infoString = f"Noise type: Additive i.i.d. noise with mean zero and signal to noise ratio: {SNR}"

