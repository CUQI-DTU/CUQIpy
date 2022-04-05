import cuqi
import numpy as np
import warnings
from scipy.interpolate import interp2d
import scipy.io as io

import astra

#=============================================================================
class ParBeamCT_2D(cuqi.problem.BayesianProblem):
    """
    2D parallel-beam Computed Tomography test problem using ASTRA

    Parameters
    ------------
    beam_type : string 
        'parallel' - 2D parallel-beam tomography,
        
    proj_type : string
        'line' - line model projection (Siddon)
        'linear' - linear interpolation projection
        'strip' - strip/area-weidght projection
    
    im_size : tuple
        Dimensions of image in pixels, default (45,45).
    
    det_count : int
        Number of detector elements, default 50.
    
    det_spacing : int
        detector element size/spacing, default 1.
    
    angles : ndarray
        Angles of projections, in radians, 
        default np.linspace(0,np.pi,60).

    noise_type : string
        The type of noise
        "Gaussian" - Gaussian white noise¨
        "scaledGaussian" - Scaled (by data) Gaussian noise

    noise_std : scalar
        Standard deviation of the noise

    prior : cuqi.distribution.Distribution
        Distribution of the prior

    Attributes
    ----------
    data : ndarray
        Generated (noisy) data

    model : cuqi.model.Model
        Deconvolution forward model

    noise : cuqi.distribution.Distribution
        Distribution of the additive noise

    prior : cuqi.distribution.Distribution
        Distribution of the prior (Default = None)

    likelihood : cuqi.likelihood.Likelihood
        Likelihood function. 
        (automatically computed from data distribution)

    exactSolution : ndarray
        Exact solution (ground truth)

    exactData : ndarray
        Noise free data

    Methods
    ----------
    MAP()
        Compute MAP estimate of posterior.
        NB: Requires prior to be defined.

    Sample(Ns)
        Sample Ns samples of the posterior.
        NB: Requires prior to be defined.


    """
    def __init__(self,
        proj_type = "linear",
        im_size=(45,45),
        det_count=50,
        det_spacing=1,
        angles=np.linspace(0,np.pi,60),
        domain = None,
        noise_type="gaussian",
        noise_cov=0.05,
        prior=None,
        data=None
        ):
        
        #######
        model = cuqi.astra.model.CT2D_parallel(proj_type = proj_type,
                                im_size = im_size,
                                det_count = det_count,
                                det_spacing = det_spacing,
                                angles = angles,
                                domain = domain) #CT model with default values
                            
        # %%
        # Extract parameters from model
        N   = model.domain_geometry.shape[0]
        p,q = model.range_geometry.shape
        n   = model.domain_geometry.dim #N*N
        m   = model.range_geometry.dim  #p*q

        # Get exact phantom
        x_exact = io.loadmat("../demos/data/phantom512.mat")["X"]
        
        #Resize phantom and make into vector
        x_exact_f = interp2d(np.linspace(0,1,x_exact.shape[0]),np.linspace(0,1,x_exact.shape[1]), x_exact)
        x_exact = x_exact_f(np.linspace(0,1,N),np.linspace(0,1,N))

        x_exact = cuqi.samples.CUQIarray(x_exact, is_par=False, geometry=model.domain_geometry)

        # Generate exact data
        b_exact = model.forward(x_exact)

        # Define and add noise #TODO: Add Poisson and logpoisson
        if noise_type.lower() == "gaussian":
            data_dist = cuqi.distribution.GaussianCov(model, noise_cov, geometry = model.range_geometry)
        elif noise_type.lower() == "scaledgaussian":
            data_dist = cuqi.distribution.GaussianCov(model, b_exact*noise_cov, geometry = model.range_geometry)
        else:
            raise NotImplementedError("This noise type is not implemented")
        
        # Generate data
        if data is None:
            data = data_dist(x_exact).sample()

        # Make likelihood
        likelihood = data_dist.to_likelihood(data)

        # Initialize CT as BayesianProblem problem
        super().__init__(likelihood, prior)

        # Store exact values
        self.exactSolution = x_exact
        self.exactData = b_exact
        self.infoString = "Noise type: Additive {} with cov: {}".format(noise_type.capitalize(),noise_cov)