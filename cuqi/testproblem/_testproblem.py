import numpy as np
from scipy.linalg import toeplitz
from scipy.sparse import csc_matrix
from scipy.integrate import quad_vec
from scipy.signal import fftconvolve
from scipy.ndimage import convolve1d

import cuqi
from cuqi.model import LinearModel
from cuqi.distribution import Gaussian
from cuqi.problem import BayesianProblem
from cuqi.geometry import Geometry, MappedGeometry, StepExpansion, KLExpansion, KLExpansion_Full, CustomKL, Continuous1D, Continuous2D, Image2D
from cuqi.array import CUQIarray
import warnings



#=============================================================================
class _Deblur(BayesianProblem):
    """
    1D Deblur test problem.

    Parameters
    ------------
    dim : int, default 128
        size of the (dim,dim) deblur problem.

    bounds : len=2 list of int's, default [0,1]
        Lower and upper bounds for the mesh.

    blur_size : int, default 48
        size of blur.

    noise_std : scalar, default 0.1
        Standard deviation of the noise.

    prior : cuqi.distribution.Distribution, default Gaussian
        Distribution of the prior.

    Attributes
    ----------
    data : ndarray
        Generated (noisy) data

    model : cuqi.model.Model
        Deblur forward model

    likelihood : cuqi.likelihood.Likelihood
        Likelihood function.

    prior : cuqi.distribution.Distribution, Default Gaussian
        Distribution of the prior

    exactSolution : ndarray
        Exact solution (ground truth)

    exactData : ndarray
        Noise free data

    mesh : ndarray
        The mesh the model is defined on.

    meshsize : float
        Size of each mesh element.

    Methods
    ----------
    MAP()
        Compute MAP estimate of posterior.
        NB: Requires prior to be defined.

    sample_posterior(Ns)
        Sample Ns samples of the posterior.
        NB: Requires prior to be defined.

    """
    def __init__(self, dim=128, bounds=[0, 1], blur_size=48, noise_std=0.1, prior=None):
        
        warnings.warn("DEPRECATED: Use Deconvolution1D instead.")
        
        # mesh
        mesh = np.linspace(bounds[0], bounds[1], dim)
        meshsize = mesh[1] - mesh[0]

        # set-up computational model kernel
        kernel = lambda x, y, blur_size: blur_size / 2*np.exp(-blur_size*abs((x-y)))   # blurring kernel

        # convolution matrix
        T1, T2 = np.meshgrid(mesh, mesh)
        A = meshsize*kernel(T1, T2, blur_size)
        maxval = A.max()
        A[A < 5e-3*maxval] = 0
        A = csc_matrix(A)   # make A sparse

        # Store forward model
        model = LinearModel(A)
        
        # Prior
        if prior is None:
            prior = Gaussian(np.zeros(dim), 1, name="x")

        # Store data distribution
        data_dist = Gaussian(model(prior), noise_std**2, name="y")
        
        # Generate inverse-crime free data (still same blur size)
        data, f_true, g_true = self._generateData(mesh, kernel, blur_size, data_dist)

        # Likelihood
        likelihood = data_dist.to_likelihood(data)

        #Initialize deblur as BayesianProblem cuqi problem
        super().__init__(likelihood, prior)
        
        #Store other properties
        self.meshsize = meshsize
        self.exactSolution = f_true
        self.exactData = g_true
        self.mesh = mesh
        
    def _generateData(self,mesh,kernel,blur_size,data_dist):

        # f is piecewise constant
        x_min, x_max = mesh[0], mesh[-1]
        vals = np.array([0, 2, 3, 2, 0, 1, 0])
        conds = lambda x: [(x_min <= x) & (x < 0.1), (0.1 <= x) & (x < 0.15), (0.15 <= x) & (x < 0.2),  \
                (0.20  <= x) & (x < 0.25), (0.25 <= x) & (x < 0.3), (0.3 <= x) & (x < 0.6), \
                (0.6 <= x) & (x <= x_max)]
        f_signal = lambda x: np.piecewise(x, conds(x), vals)

        # numerically integrate the convolution
        g_conv = lambda x: quad_vec(lambda y: f_signal(y)*kernel(x, y, blur_size), x_min, x_max)
        # se also np.convolve(kernel(...), f_true, mode='same')

        # true values
        f_true = f_signal(mesh)
        g_true = g_conv(mesh)[0]

        # noisy data
        data = g_true + data_dist(np.zeros(len(mesh))).sample() #np.squeeze(noise.sample(1)) #np.random.normal(loc=0, scale=self.sigma_obs, size=(self.dim))

        return data, f_true, g_true

#=============================================================================
class Deconvolution1D(BayesianProblem):
    """
    
    Create a 1D periodic deconvolution test problem defined by the inverse problem

    .. math::

        \mathbf{b} = \mathbf{A}\mathbf{x},

    where :math:`\mathbf{b}` is a (noisy) convolved signal,
    :math:`\mathbf{x}` is a sharp (clean) signal and
    :math:`\mathbf{A}` is a convolution operator.

    The convolution operator is defined by specifying a point spread function and
    boundary conditions and is computed via scipy.ndimage.convolve1D. By default,
    the matrix representation of the convolution operator is computed and stored.

    The inputs are padded to fit the boundary conditions.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.convolve1d.html

    Parameters
    ------------
    dim : int, default 128
        size of the (dim,dim) deconvolution problem

    PSF : string or ndarray, default 'Gauss'
        | Determines type of the underlying point spread function (PSF).
        | Depending if use_legacy is True or False, the following options are available:
        | 'Gauss' - a 1D Gaussian blur function
        | 'Moffat' - a 1D Moffat blur function (non-LEGACY only)
        | 'Defocus' - an out-of-focus 1D blur function (non-LEGACY only)
        | 'sinc' or 'prolate' - a sinc function (LEGACY only)
        | 'vonMises' - a periodic version of the Gauss function (LEGACY only)
        | ndarray - a custom PSF represented as a 1D ndarray.

    PSF_param : scalar, default depends on PSF
        | A parameter that determines the shape of the PSF;
        | the larger the parameter, the larger the blur on the signal.
        | Ignored if PSF is a ndarray.

    PSF_size : int, default equal to dim
        | The size of the PSF.
        | Ignored if PSF is a ndarray.

    BC : string, default 'periodic'
        | Boundary conditions for the convolution.
        | 'zero' - zero boundary conditions
        | 'periodic' - periodic boundary conditions
        | 'Mirror' - Reflected around center of last pixel boundary conditions
        | 'Reflect' - Reflected around edge of last pixel boundary conditions
        | 'Nearest' - Replicates last element of boundary

    phantom : string or ndarray, default 'sinc'
        | The phantom that is sampled to produce the exact solution (signal).
        | 'Gauss' - a Gaussian function
        | 'sinc' - a sinc function
        | 'vonMises' - a periodic version of the Gauss function
        | 'square' - a "top hat" function
        | 'hat' - a triangular hat function
        | 'bumps' - two bumps
        | 'derivGauss' - the first derivative of Gauss function
        | 'pc' - Piece-wise constant phantom
        | 'skyscraper' - Piece-wise constant phantom with multiple peaks
        | ndarray - a custom phantom

    phantom_param : scalar, default depends on phantom
        | A parameter that determines the horizontal scaling of the
        | function; the larger the parameter the more horizontally
        | compressed. Does not apply to phantom = 'bumps', 'pc', 'skyscraper' or ndarray.

    noise_type : string, default 'gaussian'
        | The type of noise
        | "Gaussian" - Additive Gaussian white noise¨
        | "scaledGaussian" - Gaussian noise with standard deviation
        |                    scaled by magnitude of the data for each
        |                    point.

    noise_std : scalar, default 0.01
        Standard deviation of the noise

    prior : cuqi.distribution.Distribution, Default Gaussian
        Distribution of the prior

    use_legacy : bool, Default False
        If True, use the legacy matrix representation of the forward model.
        The legacy representation has different choices for the PSF.
        The data is scaled differently than the non-legacy representation.
        

    """
    def __init__(self,
        dim=128,
        PSF="gauss",
        PSF_param=None,
        PSF_size=None,
        BC="periodic",
        phantom="sinc",
        phantom_param=None,
        noise_type="gaussian",
        noise_std=0.01,
        prior=None,
        use_legacy=False,
        ):
        
        # Set up forward model
        if use_legacy: # Legacy matrix representation

            if BC != "periodic":
                raise ValueError("Legacy matrix representation only supports periodic boundary conditions")

            if PSF_size is not None:
                raise ValueError("Legacy matrix representation does not support PSF_size")

            A = _getCirculantMatrix(dim, PSF, PSF_param)

            model = cuqi.model.LinearModel(A, range_geometry=Continuous1D(dim), domain_geometry=Continuous1D(dim))

        else: # New convolution operator based on scipy.ndimage.convolve1d

            Afun = _getConvolutionOperator(dim, PSF, PSF_param, PSF_size, BC)

            # For efficiency, we create a sparse matrix representation of the forward model,
            # instead of using the matrix-free representation. For 1D problems, there is
            # no need to use the matrix-free representation, since the matrix is small.
            Id = np.eye(dim)
            A = np.array([Afun(Id[:, i]) for i in range(dim)])
            A = csc_matrix(A) # make it sparse

            model = cuqi.model.LinearModel(A, range_geometry=Continuous1D(dim), domain_geometry=Continuous1D(dim))  

        # Set up exact solution
        if isinstance(phantom, np.ndarray):
            if phantom.ndim != 1 or phantom.shape[0] != dim:
                raise ValueError("phantom must be a 1D array of length dim")
            if phantom_param is not None:
                warnings.warn("phantom_param is ignored when phantom is a ndarray")
            x_exact = phantom
        elif isinstance(phantom, str):
            x_exact = _getExactSolution(dim, phantom, phantom_param)
        else:
            raise ValueError(f"Unknown phantom type {phantom}")

        x_exact = CUQIarray(x_exact, geometry=model.domain_geometry)

        # Generate exact data
        y_exact = model.forward(x_exact)

        # Set up prior
        if prior is None:
            prior = cuqi.distribution.Gaussian(np.zeros(dim), 1, name="x")

        # Define and add noise #TODO: Add Poisson and logpoisson
        if noise_type.lower() == "gaussian":
            data_dist = cuqi.distribution.Gaussian(model(prior), noise_std**2, name="y")
        elif noise_type.lower() == "scaledgaussian":
            data_dist = cuqi.distribution.Gaussian(model(prior), (y_exact*noise_std)**2, name="y")
        else:
            raise NotImplementedError("This noise type is not implemented")
        
        # Generate data
        data = data_dist(x_exact).sample()

        # Likelihood
        likelihood = data_dist.to_likelihood(data)

        # Set up as Bayesian problem
        super().__init__(likelihood, prior)

        # Store exact values
        self.exactSolution = x_exact
        self.exactData = y_exact
        self.infoString = "Noise type: Additive {} with std: {}".format(noise_type.capitalize(),noise_std)

def _getConvolutionOperator(dim, PSF, PSF_param, PSF_size, BC):

    # Boundary condition translation
    if BC.lower() == "zero":
        mode = "constant"
    elif BC.lower() == "periodic":
        mode = "wrap"
    elif BC.lower() == "mirror":
        mode = "mirror"
    elif BC.lower() == "reflect":
        mode = "reflect"
    elif BC.lower() == "nearest":
        mode = "nearest"
    else:
        raise ValueError("Unknown boundary condition")

    if PSF_size is None:
        PSF_size = dim
   
    # PSF setup
    if isinstance(PSF, np.ndarray):
        if PSF.ndim != 1:
            raise ValueError("PSF must be a 1D array")
        P = PSF
    elif isinstance(PSF, str):
        if PSF.lower() == "gauss":
            P, _ = _GaussPSF_1D(PSF_size, PSF_param)
        elif PSF.lower() == "moffat":
            P, _ = _MoffatPSF_1D(PSF_size, PSF_param)
        elif PSF.lower() == "defocus":
            P, _ = _DefocusPSF_1D(PSF_size, PSF_param)
        else:
            raise ValueError("Unknown PSF type")
    
    # Convolution matrix
    return lambda x: convolve1d(x, P, mode=mode)


def _createPSF_1D(PSF_size, PSF_func):
    """ Create a 1D normalized PSF of size PSF_size using the function PSF_func. """
    # Set up grid points
    x = np.arange(-np.fix(PSF_size/2), np.ceil(PSF_size/2))

    # Compute the PSF
    PSF = PSF_func(x)

    # Normalize the PSF.
    PSF /= PSF.sum()

    # find the center
    center = np.where(PSF == PSF.max())[0][0]
    return PSF, center.astype(int)

def _GaussPSF_1D(PSF_size, PSF_param):
    """ Create a 1D normalized Gaussian PSF of size PSF_size with standard deviation PSF_param. """

    # Set default value for PSF_param
    if PSF_param is None:
        PSF_param = 10

    # Set up Gaussian function
    PSF_func = lambda x: np.exp( -0.5*((x**2)/(PSF_param**2)) )

    return _createPSF_1D(PSF_size, PSF_func)

def _MoffatPSF_1D(PSF_size, PSF_param, beta=1):
    """ Create a 1D normalized Moffat PSF of size PSF_size with standard deviation PSF_param. """

    # Set default value for PSF_param
    if PSF_param is None:
        PSF_param = 10

    # Set up Moffat function
    PSF_func = lambda x: ( 1 + (x**2)/(PSF_param**2) )**(-beta)

    return _createPSF_1D(PSF_size, PSF_func)

def _DefocusPSF_1D(PSF_size, PSF_param):
    """ Create a 1D normalized defocus PSF of size PSF_size with standard deviation PSF_param. """

    # Set default value for PSF_param
    if PSF_param is None:
        PSF_param = 10

    center = np.fix(int(PSF_size/2))
    if (PSF_param == 0):    
        # the PSF is a delta function and so the blurring matrix is I
        PSF = np.zeros(PSF_size)
        PSF[center] = 1
    else:
        PSF = np.ones(PSF_size) / (np.pi * PSF_param**2)
        k = np.arange(1, PSF_size+1)
        aa = (k-center)**2
        idx = np.array((aa > (PSF_param**2)))
        PSF[idx] = 0
    PSF = PSF / PSF.sum()

    return PSF, center.astype(int)

def _getCirculantMatrix(dim, PSF, PSF_param):
    """
    GetCircMatrix  Create a circulant matrix for deconvolution examples
    
    _getCirculantMatrix(dim, PSF, PSF_param)
    
    Input:  dim = size of the circulant matrix A
            PSF = string that determined type of the underlying PSF
                    'Gauss' - a Gaussian function
                    'sinc' or 'prolate' - a sinc function
                    'vonMises' - a periodic version of the Gauss function
                    ndarray - a custom PSF.
            PSF_param = a parameter that determines the shape of the PSF;
                    the larger the parameter, to slower the initial decay
                    of the singular values of A

    Output: circulant matrix

    Based on Matlab code by Per Christian Hansen, DTU Compute, April 7, 2021
    """

    if not (dim % 2) == 0:
        raise NotImplementedError("Circulant matrix not implemented for odd numbers")

    if isinstance(PSF, np.ndarray):
        if PSF_param is not None: warnings.warn("PSF_param is ignored when PSF is a ndarray")
        if PSF.ndim != 1 or PSF.shape[0] != dim:
            raise ValueError("kernel must be a 1D array of length dim")
        h = np.roll(PSF, -int(dim/2))
        #h = h/np.linalg.norm(h)**2 # TODO: Normalize
        hflip = np.concatenate((h[0:1], np.flipud(h[1:])))
        return toeplitz(hflip,h) 

    dim_half = dim/2
    grid = np.arange(dim_half+1)/dim

    if PSF.lower() == "gauss":
        if PSF_param is None: PSF_param = 10
        h = np.exp(-(PSF_param*grid)**2)
        h = np.concatenate((h,np.flipud(h[1:-1])))
        hflip = np.concatenate((h[0:1], np.flipud(h[1:])))
        return toeplitz(hflip,h)    

    elif PSF.lower() == "sinc" or PSF.lower() == "prolate":
        if PSF_param is None: PSF_param = 15
        h = np.sinc(PSF_param*grid)
        h = np.concatenate((h,np.flipud(h[1:-1])))
        hflip = np.concatenate((h[0:1], np.flipud(h[1:])))
        return toeplitz(hflip,h)

    elif PSF.lower() == "vonmises":
        if PSF_param is None: PSF_param = 5
        h = np.exp(np.cos(2*np.pi*grid))
        h = (h/h[0])**PSF_param
        h = np.concatenate((h,np.flipud(h[1:-1])))
        hflip = np.concatenate((h[0:1], np.flipud(h[1:])))
        return toeplitz(hflip,h)

    else:
        raise NotImplementedError(f"PSF {PSF.lower()} is not implemented for the legacy convolution operator.")

def _getExactSolution(dim, phantom, phantom_param):
    """
    GetExactSolution  Create a periodic solution
    
     Input: dim = length of the vector x
            phantom = the phantom that is sampled to produce x
                    'Gauss' - a Gaussian function
                    'sinc' - a sinc function
                    'vonMises' - a periodic version of the Gauss function
                    'square' - a "top hat" function
                    'hat' - a triangular hat function
                    'bumps' - two bumps
                    'derivGauss' - the first derivative of Gauss function
                    'pc' - Piece-wise constant phantom with one tall peak and one wide peak
                    'skyscraper' - Piece-wise constant phantom with multiple peaks of varying height
            param = A parameter that determines the horizontal scaling of the
                    function; the larger the parameter the more horizontally
                    compressed. Does not apply to phantom = 'bumps' or ndarray.
    
    Output: x = column vector, scaled such that max(x) = 1

    Based on Matlab code by Per Christian Hansen, DTU Compute, April 7, 2021
    """

    if phantom.lower() == "gauss":
        if phantom_param is None: phantom_param = 5
        return np.exp(-(phantom_param*np.linspace(-1,1,dim))**2)

    elif phantom.lower() == "sinc":
        if phantom_param is None: phantom_param = 5
        return np.sinc(phantom_param*np.linspace(-1,1,dim))        

    elif phantom.lower() == "vonmises":
        if phantom_param is None: phantom_param = 5
        x = np.exp(np.cos(np.pi*np.linspace(-1,1,dim)))
        return (x/np.max(x))**phantom_param

    elif phantom.lower() == "square":
        if phantom_param is None: phantom_param = 15
        if phantom_param < 3: raise ValueError("The 'square' phantom_param must be larger than or equal to 3")
        x = np.zeros(dim)
        dimh = int(np.round(dim/2))
        w = int(np.round(dim/phantom_param))
        x[(dimh-w):(dimh+w)] = 1
        return x

    elif phantom.lower() == "hat":
        if phantom_param is None: phantom_param = 15
        if phantom_param < 3: raise ValueError("The 'hat' phantom_param must be larger than or equal to 3")
        x = np.zeros(dim)
        dimh = int(np.round(dim/2))
        w = int(np.round(dim/phantom_param))
        x[(dimh-w-1):(dimh)] = np.arange(w+1)/w
        x[(dimh-1):(dimh+w)] = np.flipud(np.arange(w+1))/w
        return x

    elif phantom.lower() == "bumps":
        if phantom_param is not None: warnings.warn("phantom_param is not used for phantom = 'bumps'")
        h = np.pi/dim
        a1 =   1; c1 = 12; t1 =  0.8
        a2 = 0.5; c2 =  5; t2 = -0.5
        grid = np.linspace(0.5,dim-0.5,dim)
        x = a1*np.exp(-c1*(-np.pi/2 + grid*h - t1)**2)+a2*np.exp(-c2*(-np.pi/2 + grid*h - t2)**2)
        return x

    elif phantom.lower() == "derivgauss":
        if phantom_param is None: phantom_param = 5
        x = np.diff(_getExactSolution(dim+1,'gauss',phantom_param))
        return x/np.max(x)
    
    elif phantom.lower() == 'pc':
        if phantom_param is not None: warnings.warn("phantom_param is not used for phantom = 'pc'")
        mesh = np.linspace(0,1,dim)
        x_min, x_max = mesh[0], mesh[-1]
        vals = np.array([0, 2, 3, 2, 0, 1, 0])
        conds = lambda x: [(x_min <= x) & (x < 0.1), (0.1 <= x) & (x < 0.15), (0.15 <= x) & (x < 0.2),  \
                (0.20  <= x) & (x < 0.25), (0.25 <= x) & (x < 0.3), (0.3 <= x) & (x < 0.6), \
                (0.6 <= x) & (x <= x_max)]
        f_signal = lambda x: np.piecewise(x, conds(x), vals)
        x = f_signal(mesh)
        return x

    elif phantom.lower() == 'skyscraper':
        if phantom_param is not None: warnings.warn("phantom_param is not used for phantom = 'skyscraper'")
        x_min = 0 
        x_max = 1
        vals = np.array([0, 1.5, 0, 1.3, 0, 0.75, 0, 0.25, 0, 1, 0])
        conds = lambda x: [(x_min <= x) & (x < 0.10), (0.10 <= x) & (x < 0.15), (0.15 <= x) & (x < 0.20),  \
                        (0.20  <= x) & (x < 0.25), (0.25 <= x) & (x < 0.35), (0.35 <= x) & (x < 0.38),\
                        (0.38  <= x) & (x < 0.45), (0.45 <= x) & (x < 0.55), \
                        (0.55  <= x) & (x < 0.75), (0.75 <= x) & (x < 0.8), (0.8 <= x) & (x <= x_max)]
        f_fun = lambda x: np.piecewise(x, conds(x), vals)
        mesh = np.linspace(x_min, x_max, dim)
        x = f_fun(mesh)
        return x


    else:
        raise NotImplementedError("This phantom is not implemented")


class Poisson1D(BayesianProblem):
    """
    1D Poisson test problem. Discretized 1D Poisson equation (steady-state linear PDE).

    Parameters
    ------------
    dim : int
        | size of the grid for the poisson problem

    endpoint : float
        | Location of end-point of grid.
    
    source : lambda function
        | Function for source term.

    field_type : str or cuqi.geometry.Geometry
        | Field type of domain. The accepted values are:
        | a Geometry object.
        | "KL": a :class:`cuqi.geometry.KLExpansion` geometry object will be created and set as a domain geometry.
        | "KL_Full": a :class:`cuqi.geometry.KLExpansion_Full` geometry object will be created and set as a domain geometry.
        | "Step": a :class:`cuqi.geometry.StepExpansion` geometry object will be created and set as a domain geometry.
        | "CustomKL": a :class:`cuqi.geometry.CustomKL` geometry object will be created and set as a domain geometry.
        | None: a :class:`cuqi.geometry.Continuous1D` geometry object will be created and set as a domain geometry.

    field_params : dict
        | A dictionary of key word arguments that the underlying geometry accepts. (Passed to the underlying geometry when field type is `KL`, `KL_Full`, `CustomKL`, `Step`). For example, for `Step` field type, the dictionary can be `{"n_steps": 3}`.

    map : lambda function
        | Mapping used to modify field.

    imap : lambda function
        | Inverse of KL map.

    SNR : int
        | Signal-to-noise ratio


    observation_grid_map : lambda function
        | Function that takes the grid as input and returns a sub-grid of the nodes where observations are available, e.g. `observation_grid_map = lambda x: x[np.where(x>5.0)]`. 

    Attributes
    ----------
    data : ndarray
        Generated (noisy) data

    model : cuqi.model.PDEModel
        Poisson 1D model

    prior : cuqi.distribution.Distribution
        Distribution of the prior

    likelihood : cuqi.likelihood.Likelihood
        Likelihood function

    exactSolution : ndarray
        Exact solution (ground truth)

    exactData : ndarray
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
    def __init__(self, dim=128, endpoint=1, source=lambda xs: 10*np.exp( -( (xs - 0.5)**2 ) / 0.02), field_type=None, field_params=None, map=None, imap=None, SNR=200, observation_grid_map=None, exactSolution=None):

        # Prepare PDE form
        N = dim-1   # Number of solution nodes
        dx = endpoint/N   # step size
        grid = np.linspace(dx, endpoint, N, endpoint=False)
        Dx = - np.diag(np.ones(N), 0) + np.diag(np.ones(N-1), 1) #Dx
        vec = np.zeros(N)
        vec[0] = 1
        Dx = np.concatenate([vec.reshape([1, -1]), Dx], axis=0)
        Dx /= dx # FD derivative matrix
        rhs = source(grid)
        
        # Grids for model
        grid_domain = np.linspace(0, endpoint, dim, endpoint=True)
        grid_range  = np.linspace(1./(dim-1), endpoint, dim-1, endpoint=False)

        # PDE form: LHS(x)u=rhs(x)
        grid_obs = grid_range
        if observation_grid_map is not None:
            grid_obs = observation_grid_map(grid_range)
        PDE_form = lambda x: (Dx.T @ np.diag(x) @ Dx, rhs)
        PDE = cuqi.pde.SteadyStateLinearPDE(PDE_form, grid_sol=grid_range,  grid_obs=grid_obs)

        # Set up geometries for model
        if field_params is None:
            field_params = {}
        if isinstance(field_type,Geometry):
            domain_geometry = field_type
        elif field_type=="KL":
            domain_geometry = KLExpansion(grid_domain, **field_params)
        elif field_type=="KL_Full":
            domain_geometry = KLExpansion_Full(grid_domain, **field_params)
        elif field_type=="Step":
            domain_geometry = StepExpansion(grid_domain, **field_params)
        elif field_type=="CustomKL":
            domain_geometry = CustomKL(grid_domain, **field_params)
        else:
            domain_geometry = Continuous1D(grid_domain)

        if map is not None:
            domain_geometry = MappedGeometry(domain_geometry, map, imap)

        range_geometry = Continuous1D(grid_obs)

        # Prepare model
        model = cuqi.model.PDEModel(PDE,range_geometry,domain_geometry)

        # Set up exact solution
        if exactSolution is None:
            x_exact = np.exp( 5*grid_domain*np.exp(-2*grid_domain)*np.sin(endpoint-grid_domain) ) 
        else: 
            x_exact = exactSolution 
        x_exact = CUQIarray(x_exact, is_par=False, geometry=domain_geometry)

        # Generate exact data
        y_exact = model.forward(x_exact,is_par=False)

        # Add noise to data
        sigma = np.linalg.norm(y_exact)/SNR
        sigma2 = sigma*sigma # variance of the observation Gaussian noise
        data = y_exact + np.random.normal(0, sigma, y_exact.shape)

        # Bayesian model
        x = cuqi.distribution.Gaussian(np.zeros(model.domain_dim), 1)
        y = cuqi.distribution.Gaussian(model, sigma2)

        # Initialize Deconvolution as BayesianProblem problem
        super().__init__(y, x, y=data)

        # Store exact values
        self.exactSolution = x_exact
        self.exactData = y_exact

class Heat1D(BayesianProblem):
    """
    1D Heat test problem. Discretized Heat equation (time-dependent linear PDE).

    Parameters
    ------------
    dim : int
        | size of the grid for the heat problem

    endpoint : float
        | Location of end-point of grid.
    
    max_time : float
        | The last time step.
    
    field_type : str or cuqi.geometry.Geometry
        | Field type of domain. The accepted values are:
        | a Geometry object.
        | "KL": a :class:`cuqi.geometry.KLExpansion` geometry object will be created and set as a domain geometry.
        | "KL_Full": a :class:`cuqi.geometry.KLExpansion_Full` geometry object will be created and set as a domain geometry.
        | "Step": a :class:`cuqi.geometry.StepExpansion` geometry object will be created and set as a domain geometry.
        | "CustomKL": a :class:`cuqi.geometry.CustomKL` geometry object will be created and set as a domain geometry.
        | None: a :class:`cuqi.geometry.Continuous1D` geometry object will be created and set as a domain geometry.

    field_params : dict
        | A dictionary of key word arguments that the underlying geometry accepts. (Passed to the underlying geometry when field type is `KL`, `KL_Full`, `CustomKL`, `Step`). For example, for `Step` field type, the dictionary can be `{"n_steps": 3}`.

    map : lambda function
        | If given, an underlying `MappedGeometry` geometry object is created which applies the mapping on the field, e.g. for log parameterization: `map = lambda x:np.exp(x)`.

    imap : lambda function
        | The inverse of the provided map. 

    SNR : int
        | Signal-to-noise ratio

    exactSolution : ndarray or CUQIarray
        | The exact solution of the problem which is the heat model initial condition in this test problem. If provided as None, an exact solution is generated, if provided as ndarray, it is assumed to be function values and it is converted to a CUQIarray.


    observation_grid_map : lambda function
       | Function that takes the grid as input and returns a sub-grid of the nodes where observations are available, e.g. `observation_grid_map = lambda x: x[np.where(x>5.0)]`. 
 
    Attributes
    ----------
    data : ndarray
        Generated (noisy) data

    model : cuqi.model.PDEModel
        Heat 1D model

    prior : cuqi.distribution.Distribution
        Distribution of the prior

    likelihood : cuqi.likelihood.Likelihood
        Likelihood function

    exactSolution : ndarray
        Exact solution (ground truth)

    exactData : ndarray
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
    def __init__(self, dim=128, endpoint=1, max_time=0.2, field_type=None, field_params=None,map=None, imap=None, SNR=200, exactSolution=None, observation_grid_map=None):
        

        # Prepare PDE form
        N = dim   # Number of solution nodes
        dx = endpoint/(N+1)   # space step size
        cfl = 5/11 # the cfl condition to have a stable solution
        dt_approx = cfl*dx**2 # defining approximate time step size
        max_iter = int(max_time/dt_approx) # number of time steps
        Dxx = (np.diag( -2*np.ones(N) ) + np.diag(np.ones(N-1),-1) + np.diag(np.ones(N-1),1))/dx**2 # FD diffusion operator
        
        # Grids for model
        grid_domain = np.linspace(dx, endpoint, N, endpoint=False)
        grid_range = np.linspace(dx, endpoint, N, endpoint=False) 
        time_steps = np.linspace(0,max_time,max_iter+1,endpoint=True)

        # PDE form (diff_op, IC, time_steps)
        grid_obs = grid_range
        if observation_grid_map is not None:
            grid_obs = observation_grid_map(grid_obs)

        def PDE_form(IC, t): return (Dxx, np.zeros(N), IC)
        PDE = cuqi.pde.TimeDependentLinearPDE(
            PDE_form, time_steps, grid_sol=grid_domain, grid_obs=grid_obs)

        # Set up geometries for model
        if field_params is None:
            field_params = {}
        if isinstance(field_type,Geometry):
            domain_geometry = field_type
        elif field_type=="KL":
            domain_geometry = KLExpansion(grid_domain, **field_params)
        elif field_type=="KL_Full":
            domain_geometry = KLExpansion_Full(grid_domain, **field_params)
        elif field_type=="Step":
            domain_geometry = StepExpansion(grid_domain, **field_params)
        elif field_type=="CustomKL":
            domain_geometry = CustomKL(grid_domain, **field_params)
        else:
            domain_geometry = Continuous1D(grid_domain)
        if map is not None:
            domain_geometry = MappedGeometry(domain_geometry,map,imap)
        range_geometry = Continuous1D(grid_obs)

        # Prepare model
        model = cuqi.model.PDEModel(PDE,range_geometry,domain_geometry)
        if exactSolution is not None:
            x_exact = CUQIarray(exactSolution, is_par = False, geometry=domain_geometry)
        # Set up exact solution
        
        else:
            if field_type=="Step":
                n_steps = domain_geometry.n_steps
                x_exact = CUQIarray(domain_geometry.par2fun(np.array(range(n_steps))), is_par=False, geometry=domain_geometry)
            else:
                grid_domain = model.domain_geometry.grid
                x_exact = grid_domain*np.exp(-2*grid_domain)*np.sin(endpoint-grid_domain)
                x_exact = CUQIarray(x_exact, is_par=False, geometry=domain_geometry)
        #x_exact = 100*grid_domain*np.exp(-5*grid_domain)*np.sin(endpoint-grid_domain)
        # Generate exact data
        y_exact = model.forward(x_exact, is_par=False)
        # Add noise to data
        sigma = np.linalg.norm(y_exact)/SNR
        sigma2 = sigma*sigma # variance of the observation Gaussian noise
        data = y_exact + np.random.normal(0, sigma, y_exact.shape)

        # Bayesian model
        x = cuqi.distribution.Gaussian(np.zeros(model.domain_dim), 1)
        y = cuqi.distribution.Gaussian(model(x), sigma2)
        
        # Initialize Deconvolution as BayesianProblem problem
        super().__init__(y, x, y=data)

        # Store exact values
        self.exactSolution = x_exact
        self.exactData = y_exact
        self.infoString = f"Noise type: Additive i.i.d. noise with mean zero and signal to noise ratio: {SNR}"


class Abel1D(BayesianProblem):
    """
    1D Abel test problem. 1D model of rotationally symmetric computed tomography.

    Parameters
    ------------
    dim : int
        size of the grid for the problem

    endpoint : float
        Location of end-point of grid.
    
    field_type : str or cuqi.geometry.Geometry
        Field type of domain.

    KL_map : lambda function
        Mapping used to modify field.

    KL_imap : lambda function
        Inverse of KL map.

    SNR : int
        Signal-to-noise ratio

    Attributes
    ----------
    data : ndarray
        Generated (noisy) data

    model : cuqi.model.LinearModel
        Abel 1D model

    prior : cuqi.distribution.Distribution
        Distribution of the prior

    likelihood : cuqi.likelihood.Likelihood
        Likelihood function

    exactSolution : ndarray
        Exact solution (ground truth)

    exactData : ndarray
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
    def __init__(self, dim=128, endpoint=1, field_type=None, field_params=None, KL_map=None, KL_imap=None, SNR=100):
        N = dim # number of quadrature points
        h = endpoint/N # quadrature weight

        tvec = np.linspace(h/2, endpoint-h/2, N).reshape(1, -1) 
        svec = tvec.reshape(-1, 1) + h/2
        tmat = np.tile( tvec, [N, 1] )
        smat = np.tile( svec, [1, N] )
        
        idx = np.where(tmat<smat) # only applying the quadrature on 0<x<1
        A = np.zeros([N,N]) # Abel integral operator
        A[idx[0], idx[1]] = h/np.sqrt( np.abs( smat[idx[0], idx[1]] - tmat[idx[0], idx[1]] ) )

        # discretization
        grid = np.linspace(0, endpoint, N)

        # Geometry
        if field_params is None:
            field_params = {}
        if isinstance(field_type,Geometry):
            domain_geometry = field_type
        elif field_type=="KL":
            domain_geometry = KLExpansion(grid,**field_params)
        elif field_type=="Step":
            domain_geometry = StepExpansion(grid,**field_params)
        elif field_type=="CustomKL":
            domain_geometry = CustomKL(grid,**field_params)
        else:
            domain_geometry = Continuous1D(grid)

        if KL_map is not None:
            domain_geometry = MappedGeometry(domain_geometry,KL_map,KL_imap)

        range_geometry = Continuous1D(grid)

        # Set up model
        model = LinearModel(A,range_geometry=range_geometry, domain_geometry=domain_geometry)
    
        # Set up exact solution
        x_exact = np.sin(tvec*np.pi)*np.exp(-2*tvec)
        x_exact.shape = (dim,)
        x_exact = CUQIarray(x_exact, is_par=False, geometry=domain_geometry)

        # Generate exact data
        y_exact = model.forward(x_exact,is_par=False)

        # Add noise to data
        sigma = np.linalg.norm(y_exact)/SNR
        sigma2 = sigma*sigma # variance of the observation Gaussian noise
        data = y_exact + np.random.normal(0, sigma, y_exact.shape )

        # Bayesian model
        x = cuqi.distribution.Gaussian(np.zeros(model.domain_dim), 1)
        y = cuqi.distribution.Gaussian(model(x), sigma2)
        
        # Initialize Deconvolution as BayesianProblem problem
        super().__init__(y, x, y=data)

        # Store exact values
        self.exactSolution = x_exact
        self.exactData = y_exact


class _Deconv_1D(BayesianProblem):
    """
    1D Deconvolution test problem. Discreate linear problem from blurring kernel.

    Parameters
    ------------
    dim : int
        size of the grid for the problem

    endpoint : float
        Location of end-point of grid.
    
    field_type : str or cuqi.geometry.Geometry
        Field type of domain.

    KL_map : lambda function
        Mapping used to modify field.

    KL_imap : lambda function
        Inverse of KL map.

    SNR : int
        Signal-to-noise ratio

    Attributes
    ----------
    data : ndarray
        Generated (noisy) data

    model : cuqi.model.LinearModel
        Deconvolution 1D model

    prior : cuqi.distribution.Distribution
        Distribution of the prior

    likelihood : cuqi.likelihood.Likelihood
        Likelihood function

    exactSolution : ndarray
        Exact solution (ground truth)

    exactData : ndarray
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
    def __init__(self, dim=128, endpoint=1, kernel=None, blur_size=48, field_type=None, field_params=None, KL_map=None, KL_imap=None, SNR=100):
        
        warnings.warn("DEPRECATED: Use Deconvolution1D instead.")
        
        N = dim # number of quadrature points
        h = endpoint/N # quadrature weight
        grid = np.linspace(0, endpoint, N)

        if kernel is None:
            kernel = lambda x, y, blur_size_var: blur_size_var / 2*np.exp(-blur_size*abs((x-y)))   # blurring kernel
        
        # convolution matrix
        T1, T2 = np.meshgrid(grid, grid)
        A = h*kernel(T1, T2, blur_size)
        maxval = A.max()
        A[A < 5e-3*maxval] = 0
        A = csc_matrix(A)   # make A sparse
        
        # discretization
        if isinstance(field_type,Geometry):
            domain_geometry = field_type
        elif field_type=="KL":
            domain_geometry = KLExpansion(grid,field_params)
        elif field_type=="Step":
            domain_geometry = StepExpansion(grid)
        elif field_type=="CustomKL":
            domain_geometry = CustomKL(grid,field_params)
        else:
            domain_geometry = Continuous1D(grid)

        if KL_map is not None:
            domain_geometry = MappedGeometry(domain_geometry,KL_map,KL_imap)

        range_geometry = Continuous1D(grid)

        # Set up model
        model = LinearModel(A,range_geometry=range_geometry, domain_geometry=domain_geometry)
    
        # Prior
        prior = cuqi.distribution.Gaussian(np.zeros(model.domain_dim), 1, geometry=model.domain_geometry, name="x")

        # Set up exact solution
        x_exact = prior.sample()

        # Generate exact data
        y_exact = model.forward(x_exact)

        # Add noise to data
        sigma = np.linalg.norm(y_exact)/SNR
        sigma2 = sigma*sigma # variance of the observation Gaussian noise
        data = y_exact + np.random.normal(0, sigma, y_exact.shape)

        likelihood = cuqi.distribution.Gaussian(model(prior), sigma2, name="y").to_likelihood(data)
        
        # Initialize Deconvolution as BayesianProblem problem
        super().__init__(likelihood, prior)

        # Store exact values
        self.exactSolution = x_exact
        self.exactData = y_exact


#=============================================================================
class Deconvolution2D(BayesianProblem):
    """
    
    Create a 2D Deconvolution test problem defined by the inverse problem

    .. math::

        \mathbf{b} = \mathbf{A}\mathbf{x},

    where :math:`\mathbf{b}` is a (noisy) blurred image,
    :math:`\mathbf{x}` is a sharp image and
    :math:`\mathbf{A}` is a convolution operator.

    The convolution operator is defined by specifying a point spread function and
    boundary conditions and is computed (matrix-free) via scipy.signal.fftconvolve.
    The inputs are padded to fit the boundary conditions.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.fftconvolve.html.

    Parameters
    ----------
    dim : int
        | size of the (dim,dim) deconvolution problem

    PSF : string or ndarray
        | Determines type of the underlying point spread function (PSF).
        | 'Gauss' - a Gaussian blur
        | 'Moffat' - a Moffat blur blur
        | 'Defocus' - an out-of-focus blur
        | ndarray - Custom user-specified PSF

    PSF_param : scalar
        | A parameter that determines the shape of the PSF;
        | the larger the parameter, the larger blur on the image.
        | Ignored if PSF is given as ndarray

    PSF_size : int
        | Defines the size of the PSF. The size becomes (PSF_size, PSF_size).
        | Ignored if PSF is given as ndarray

    BC : string
        | Boundary conditions of convolution
        | 'zero' - Zero boundary
        | 'periodic' - Periodic boundary
        | 'Neumann' - Neumann (reflective) boundary
        | 'Mirror' - Mirrored boundary
        | 'Nearest' - Replicates last element of boundary
        
    phantom : string or ndarray
        | The phantom (sharp image) that is convolved.
        | If ndarray it should be a square 2D array representing an image.
        | The image will automatically be resized to fit the problem size.
        | If string it should be any 2D phantom defined in cuqi.data.
        | The string is lowercased and any hyphens are replaced 
        | with underscores to match a method name in cuqi.data.
        | Examples:
        | 'astronaut' - a photo of an astronaut.
        | 'camera' - a photo of a man with a camera.
        | 'cat' - a photo of a cat.
        | 'cookie' - cartoon cookie.
        | 'satellite' - a photo of a satellite.

    noise_type : string
        | The type of noise
        | "Gaussian" - Gaussian white noise¨
        | "scaledGaussian" - Gaussian noise where standard deviation is scaled by by data.

    noise_std : scalar
        | Standard deviation of the noise.

    prior : Distribution
        | Option to set the prior distribution.
        | If set all components of the posterior
        | are defined and sampling can be achieved.

    Attributes
    ----------
    data : CUQIarray
        Generated (noisy) data

    model : Model
        Deconvolution forward model

    prior : Distribution
        Distribution of the prior

    likelihood : Likelihood
        The likelihood function
        (automatically computed from data distribution)

    exactSolution : CUQIarray
        Exact solution (ground truth)

    exactData : CUQIarray
        Noise free data.

    Methods
    -------
    MAP()
        Compute MAP estimate of posterior.
        NB: Requires prior to be defined

    sample_posterior(Ns)
        Sample Ns samples of the posterior.
        NB: Requires prior to be defined
    """
    def __init__(self,
        dim=128,
        PSF="gauss",
        PSF_param=2.56,
        PSF_size=21,
        BC="periodic",
        phantom="satellite",
        noise_type="gaussian",
        noise_std=0.0036,
        prior = None):
        
        # setting up the geometry
        domain_geometry = Image2D((dim, dim))
        range_geometry = Image2D((dim, dim))

        # set boundary conditions
        if (BC.lower() == "neumann"):
            BC = "symmetric"
        elif (BC.lower() == "zero"):
            BC = "constant" # cval = 0
        elif (BC.lower() == "nearest"):
            BC = "edge" # the input is extended by replicating the last pixel
        elif (BC.lower() == "mirror"):
            BC = "reflect" # reflecting about the center of the last pixel
        elif (BC.lower() == "periodic"):
            BC = "wrap"
        else:
            raise TypeError("Unknown BC type")

        # Set up PSF.
        if isinstance(PSF, np.ndarray):
            P = PSF
        elif isinstance(PSF, str):
            if PSF.lower() == "gauss":
                P, _ = _GaussPSF(np.array([PSF_size, PSF_size]), PSF_param)
            elif PSF.lower() == "moffat":
                P, _ = _MoffatPSF(np.array([PSF_size, PSF_size]), PSF_param, 1)
            elif PSF.lower() == "defocus":
                P, _ = _DefocusPSF(np.array([PSF_size, PSF_size]), PSF_param)
        else:
            raise TypeError(f"Unknown PSF: {PSF}.")

        # build forward model
        model = cuqi.model.LinearModel(lambda x: _proj_forward_2D(x, P, BC), 
                                       lambda x: _proj_backward_2D(x, P, BC), 
                                       range_geometry, 
                                       domain_geometry)

        # User provided phantom as ndarray
        if isinstance(phantom, np.ndarray):
            if phantom.ndim > 2:
                raise ValueError("Input phantom image must be an image (matrix) or vector")
            if phantom.ndim == 1:
                N = int(round(np.sqrt(len(phantom))))
                phantom = phantom.reshape(N,N)
            phantom = cuqi.data.imresize(phantom, dim) # Resize phantom (if wrong size)
            x_exact2D = phantom
        # If phantom is string its a specific case
        elif isinstance(phantom, str):
            # lowercase and replace hyphens with underscores to match library method names
            phantom = phantom.lower().replace("-", "_") 
            if hasattr(cuqi.data, phantom):
                x_exact2D = getattr(cuqi.data, phantom)(size=dim)
            else:
                raise ValueError(f"Phantom {phantom} not found in cuqi.data phantom library.")
        else:
            raise TypeError("Unknown phantom type. Must be ndarray or string.")

        x_exact = x_exact2D.flatten()
        x_exact = CUQIarray(x_exact, is_par=True, geometry=domain_geometry)

        # Generate exact data (blurred)
        y_exact = model@x_exact

        # Create prior
        if prior is None:
            prior = cuqi.distribution.Gaussian(np.zeros(model.domain_dim), 1, geometry=domain_geometry, name="x")

        # Data distribution
        if noise_type.lower() == "gaussian":
            data_dist = cuqi.distribution.Gaussian(model(prior), noise_std**2, geometry=range_geometry, name="y")
        elif noise_type.lower() == "scaledgaussian":
            data_dist = cuqi.distribution.Gaussian(model(prior), (y_exact*noise_std)**2, geometry=range_geometry, name="y")
        else:
            raise NotImplementedError("This noise type is not implemented")
        
        # Generate noisy data
        data = data_dist(x_exact).sample()

        # Likelihood
        likelihood = data_dist.to_likelihood(data)
        
        # Initialize Deconvolution as BayesianProblem problem
        super().__init__(likelihood, prior)

        self.exactSolution = x_exact
        self.exactData = y_exact
        self.infoString = "Noise type: Additive {} with std: {}".format(noise_type.capitalize(),noise_std)
        self.Miscellaneous = {"PSF" : P, "BC": BC}

#=========================================================================
def _proj_forward_2D(X, P, BC):
    PSF_size = max(P.shape)
    X_padded = np.pad(X, PSF_size//2, mode=BC)
    Ax = fftconvolve(X_padded, P, mode='valid')
    if not PSF_size & 0x1: # If PSF_size is even
        Ax = Ax[1:, 1:] # Remove first row and column to fit convolve math
    return Ax

#=========================================================================
def _proj_backward_2D(B, P, BC):
    P = np.flipud(np.fliplr(P)) # Flip PSF
    return _proj_forward_2D(B, P, BC)

# ===================================================================
# Array with PSF for Gaussian blur (astronomic turbulence)
# ===================================================================
def _GaussPSF(dim, s):
    if hasattr(dim, "__len__"):
        m, n = dim[0], dim[1]
    else:
        m, n = dim, dim
    s1, s2 = s, s
    
    # Set up grid points to evaluate the Gaussian function
    x = np.arange(-np.fix(n/2), np.ceil(n/2))
    y = np.arange(-np.fix(m/2), np.ceil(m/2))
    X, Y = np.meshgrid(x, y)

    # Compute the Gaussian, and normalize the PSF.
    PSF = np.exp( -0.5* ((X**2)/(s1**2) + (Y**2)/(s2**2)) )
    PSF /= PSF.sum()

    # find the center
    mm, nn = np.where(PSF == PSF.max())
    center = np.array([mm[0], nn[0]])

    return PSF, center.astype(int)
# ===================================================================
# Array with PSF for Moffat blur (astronomical telescope)
# ===================================================================
def _MoffatPSF(dim, s, beta):
    if hasattr(dim, "__len__"):
        m, n = dim[0], dim[1]
    else:
        m, n = dim, dim
    s1, s2 = s, s
    
    # Set up grid points to evaluate the Gaussian function
    x = np.arange(-np.fix(n/2), np.ceil(n/2))
    y = np.arange(-np.fix(m/2), np.ceil(m/2))
    X, Y = np.meshgrid(x, y)

    # Compute the Gaussian, and normalize the PSF.
    PSF = ( 1 + (X**2)/(s1**2) + (Y**2)/(s2**2) )**(-beta)
    PSF = PSF / PSF.sum()

    # find the center
    mm, nn = np.where(PSF == PSF.max())
    center = np.array([mm[0], nn[0]])

    return PSF, center.astype(int)
# ===================================================================
# Array with PSF for out-of-focus blur
# ===================================================================
def _DefocusPSF(dim, R):
    if hasattr(dim, "__len__"):
        m, n = dim[0], dim[1]
    else:
        m, n = dim, dim
    
    center = np.fix((np.array([m, n]))/2)
    if (R == 0):    
        # the PSF is a delta function and so the blurring matrix is I
        PSF = np.zeros((m, n))
        PSF[center[0], center[1]] = 1
    else:
        PSF = np.ones((m, n)) / (np.pi * R**2)
        k = np.arange(1, max(m, n)+1)
        aa, bb = (k-center[0])**2, (k-center[1])**2
        A, B = np.meshgrid(aa, aa), np.meshgrid(bb, bb)
        idx = np.array(((A[0].T + B[0]) > (R**2)))
        PSF[idx] = 0
    PSF = PSF / PSF.sum()

    return PSF, center.astype(int)


#=============================================================================
class WangCubic(BayesianProblem):
    """ Two parameters and one observation cubic test problem.
    
    Parameters
    ------------
    noise_std : scalar
        Standard deviation of the noise

    prior : cuqi.distribution.Distribution
        Distribution of the prior
    
    data : scalar
        Observed data
    
    Notes
    -----
    Based on Section 3.3.2 in Wang (2015):
    Z. Wang, "An Optimization Based Algorithm for Bayesian Inference". Master thesis. MIT. 2015  
    https://dspace.mit.edu/bitstream/handle/1721.1/98815/921147308-MIT.pdf?sequence=1&isAllowed=y

    """
    def __init__(self, noise_std=1, prior=None, data=None):
        # forward model and gradient
        def forward(x):
            return 10*x[1] - 10*x[0]**3 + 5*x[0]**2 + 6*x[0]
        def jacobian(x):
            return np.array([[-30*x[0]**2 + 10*x[0] + 6, 10]])
        model = cuqi.model.Model(forward, range_geometry=1, domain_geometry=2, jacobian=jacobian)

        # define prior
        if prior is None:
            prior = cuqi.distribution.Gaussian(np.array([1, 0]), 1, name="x")

        # data
        if data is None:
            data = 1

        # data distribution is Gaussian
        data_dist = cuqi.distribution.Gaussian(model(prior), noise_std**2, name="y")

        # Define Gaussian likelihood
        likelihood = data_dist.to_likelihood(data)

        # Set up Bayesian Problem
        super().__init__(likelihood, prior)

        # Store exact values
        self.exactSolution = None
        self.exactData = None
        self.infoString = "Noise type: Additive {} with std: {}".format('Gaussian', noise_std)
