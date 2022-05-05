import cuqi
import numpy as np

#=============================================================================
class ParallelBeam2DProblem(cuqi.problem.BayesianProblem):
    """ 2D parallel-beam Computed Tomography test problem using ASTRA

    Parameters
    ----------
        
    im_size : tuple of ints
        Dimensions of image in pixels.
    
    det_count : int
        Number of detector elements.

    angles : ndarray
        Angles of projections, in radians.
    
    det_spacing : int, optional
        Detector element size/spacing.
        Default is handled by the CT model.

    domain : tuple of ints, optional
        Domain of the image.
        Default is handled by the CT model.

    proj_type : string
        'line' - line model projection (Siddon)
        'linear' - linear interpolation projection
        'strip' - strip/area-weight projection

    phantom : str or ndarray
        Phantom image to generate data from.
        If string name must match a phantom in cuqi.data.
        The string is lowercased and any hyphens are replaced 
        with underscores to match a method name in cuqi.data.

    noise_type : string
        The type of noise
        "Gaussian" - Gaussian white noise
        "scaledGaussian" - Scaled (by data) Gaussian noise

    noise_std : scalar
        Standard deviation of the noise

    prior : cuqi.distribution.Distribution, optional
        Distribution of the prior.
        If set posterior samples can be computed using :meth:`sample_posterior`.

    data : cuqi.samples.CUQIarray, optional
        Data to be stored in testproblem.

    Attributes
    ----------
    data : CUQIarray
        Generated (noisy) data

    model : cuqi.astra.model.ASTRAModel
        CT model.

    prior : cuqi.distribution.Distribution, Default None
        Distribution of the prior.

    likelihood : cuqi.likelihood.Likelihood
        Likelihood function. 
        (automatically computed from noise distribution)

    exactSolution : CUQIarray
        Exact solution (ground truth)

    exactData : CUQIarray
        Noise free data

    Methods
    ----------
    MAP()
        Compute MAP estimate of posterior.
        NB: Requires prior to be defined.

    sample_posterior(Ns)
        Sample Ns samples of the posterior.
        NB: Requires prior to be defined.

    """
    def __init__(self,
        im_size=(45,45),
        det_count=50,
        angles=np.linspace(0, np.pi, 60),
        det_spacing=None,
        domain=None,
        proj_type="linear",
        phantom="shepp-logan",
        noise_type="gaussian",
        noise_std=0.05,
        prior=None,
        data=None
        ):
        
        # CT model
        model = cuqi.astra.model.ParallelBeam2DModel(
            im_size=im_size,
            det_count=det_count,
            angles=angles,
            det_spacing=det_spacing,
            domain=domain,
            proj_type=proj_type
        )
                    
        # Get exact phantom
        if isinstance(phantom, np.ndarray):
            if phantom.shape != model.domain_geometry.shape:
                raise ValueError("Phantom shape does not match model domain geometry.")
            x_exact = phantom
        elif isinstance(phantom, str):
            # lowercase and replace hyphens with underscores to match library method names
            phantom = phantom.lower().replace("-", "_") 
            if hasattr(cuqi.data, phantom):
                x_exact = getattr(cuqi.data, phantom)(size=model.domain_geometry.shape[0])
            else:
                raise ValueError("Phantom not found in cuqi.data phantom library.")
        else:
            raise ValueError("Phantom must be a string or ndarray. See string options in cuqi.data.")
        
        x_exact = cuqi.samples.CUQIarray(x_exact, is_par=False, geometry=model.domain_geometry)

        # Generate exact data
        b_exact = model.forward(x_exact)

        # Define and add noise #TODO: Add Poisson and logpoisson
        if noise_type.lower() == "gaussian":
            data_dist = cuqi.distribution.GaussianCov(model, noise_std**2, geometry=model.range_geometry)
        elif noise_type.lower() == "scaledgaussian":
            data_dist = cuqi.distribution.GaussianCov(model, b_exact*(noise_std**2), geometry=model.range_geometry)
        else:
            raise NotImplementedError("This noise type is not implemented")
        
        # Generate data
        if data is None:
            b_exact = model.forward(x_exact)
            data = data_dist(x_exact).sample()
        else:
            b_exact = None # No exact data if data is provided

        # Make likelihood
        likelihood = data_dist.to_likelihood(data)

        # Initialize CT as BayesianProblem problem
        super().__init__(likelihood, prior)

        # Store exact values
        self.exactSolution = x_exact
        self.exactData = b_exact
        self.infoString = "Noise type: Additive {} with std: {}".format(noise_type.capitalize(), noise_std)