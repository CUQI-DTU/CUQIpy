import numpy as np
from scipy.linalg import toeplitz
from scipy.sparse import csc_matrix
from scipy.integrate import quad_vec

import cuqi
from cuqi.model import LinearModel
from cuqi.distribution import Gaussian
from cuqi.problem import Type1

#=============================================================================
class Deblur(Type1):
    
    def __init__(self, a = 48, noise_std = 0.1, dim = 128, bnds = [0, 1]):
        t = np.linspace(bnds[0], bnds[1], dim)
        h = t[1] - t[0]

        # set-up computational model kernel
        kernel = lambda x, y, a: a / 2*np.exp(-a*abs((x-y)))   # blurring kernel

        # convolution matrix
        T1, T2 = np.meshgrid(t, t)
        A = h*kernel(T1, T2, a)
        maxval = A.max()
        A[A < 5e-3*maxval] = 0
        A = csc_matrix(A)   # make A sparse

        # Store forward model
        model = LinearModel(A)
        
        # Store Noise model
        noise = Gaussian(np.zeros(dim),noise_std,np.eye(dim))
        
        # Generate inverse-crime free data
        data, f_true, g_true = data_conv(t,kernel,noise)
        
        #Initialize deblur as Type1 cuqi probler
        super().__init__(data,model,noise,[]) #No default prior
        
        #Store other properties
        self.meshsize = h
        self.f_true = f_true
        self.g_true = g_true
        self.t = t
        

def data_conv(t,kernel,noise):
    np.random.seed(1)

    # f is piecewise constant
    x_min, x_max = t[0], t[-1]
    vals = np.array([0, 2, 3, 2, 0, 1, 0])
    conds = lambda x: [(x_min <= x) & (x < 0.1), (0.1 <= x) & (x < 0.15), (0.15 <= x) & (x < 0.2),  \
               (0.20  <= x) & (x < 0.25), (0.25 <= x) & (x < 0.3), (0.3 <= x) & (x < 0.6), \
               (0.6 <= x) & (x <= x_max)]
    f_signal = lambda x: np.piecewise(x, conds(x), vals)

    # numerically integrate the convolution
    a_true = 50
    g_conv = lambda x: quad_vec(lambda y: f_signal(y)*kernel(x, y, a_true), x_min, x_max)
    # se also np.convolve(kernel(...), f_true, mode='same')

    # true values
    f_true = f_signal(t)
    g_true = g_conv(t)[0]

    # noisy data
    b = g_true + np.squeeze(noise.sample(1)) #np.random.normal(loc=0, scale=self.sigma_obs, size=(self.dim))

    return b, f_true, g_true

#=============================================================================
class Deconvolution(Type1):
    """
    1D Deconvolution test problem

    Parameters
    ------------
    dim : int
        size of the (dim,dim) deconvolution problem

    kernel : string 
        Determines type of the underlying kernel
        'Gauss' - a Gaussian function
        'sinc' or 'prolate' - a sinc function
        'vonMises' - a periodic version of the Gauss function

    kernel_param : scalar
        A parameter that determines the shape of the kernel;
        the larger the parameter, the slower the initial
        decay of the singular values of A

    phantom : string
        The phantom that is sampled to produce x
        'Gauss' - a Gaussian function
        'sinc' - a sinc function
        'vonMises' - a periodic version of the Gauss function
        'square' - a "top hat" function
        'hat' - a triangular hat function
        'bumps' - two bumps
        'derivGauss' - the first derivative of Gauss function

    phantom_param : scalar
        A parameter that determines the width of the central 
        "bump" of the function; the larger the parameter,
        the narrower the "bump."  
        Does not apply to phantom = 'bumps'

    noise_type : string
        The type of noise
        "Gaussian" - Gaussian white noise¨
        "xxxGaussian" - Scaled (by data) Gaussian noise

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

    likelihood : cuqi.distribution.Distribution
        Distribution of the likelihood 
        (automatically computed from noise distribution)

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
        dim=128,
        kernel="gauss",
        kernel_param=None,
        phantom="gauss",
        phantom_param=None,
        noise_type="gaussian",
        noise_std=0.05,
        prior=None,
        data=None,
        noise=None
        ):
        
        # Set up model
        A = _getCirculantMatrix(dim,kernel,kernel_param)
        model = cuqi.model.LinearModel(A)

        # Set up exact solution
        x_exact = _getExactSolution(dim,phantom,phantom_param)

        # Generate exact data
        b_exact = model.forward(x_exact)

        if noise is None:
            # Define and add noise
            if noise_type.lower() == "gaussian":
                noise = cuqi.distribution.Gaussian(np.zeros(dim),noise_std,np.eye(dim))
            elif noise_type.lower() == "xxxgaussian":
                noise = cuqi.distribution.Gaussian(np.zeros(dim),b_exact*noise_std,np.eye(dim))
            #TODO elif noise_type.lower() == "poisson":
            #TODO elif noise_type.lower() == "logpoisson":
            else:
                raise NotImplementedError("This noise type is not implemented")
        
        if data is None:
            data = b_exact + noise.sample(1).flatten()

        # Initialize Deconvolution as Type1 problem
        super().__init__(data,model,noise,prior)

        self.exactSolution = x_exact
        self.exactData = b_exact

def _getCirculantMatrix(dim,kernel,kernel_param):
    """
    GetCircMatrix  Create a circulant matrix for deconvolution examples
    
    GetCircMatrix(dim,kernel,kernel_param)
    
    Input:  dim = size of the circulant matrix A
            kernel = string that determined type of the underlying kernel
                    'Gauss' - a Gaussian function
                    'sinc' or 'prolate' - a sinc function
                    'vonMises' - a periodic version of the Gauss function
            kernel_param = a parameter that determines the shape of the kernel;
                    the larger the parameter, to slower the initial decay
                    of the singular values of A

    Output: circulant matrix

    Based on Matlab code by Per Christian Hansen, DTU Compute, April 7, 2021
    """

    if not (dim % 2) == 0:
        raise NotImplementedError("Circulant matrix not implemented for odd numbers")

    dim_half = dim/2
    grid = np.arange(dim_half+1)/dim

    if kernel.lower() == "gauss":
        if kernel_param is None: kernel_param = 10
        h = np.exp(-(kernel_param*grid)**2)
        h = np.concatenate((h,np.flipud(h[1:-1])))
        hflip = np.concatenate((h[0:1], np.flipud(h[1:])))
        return toeplitz(hflip,h)    

    elif kernel.lower() == "sinc" or kernel.lower() == "prolate":
        if kernel_param is None: kernel_param = 15
        h = np.sinc(kernel_param*grid)
        h = np.concatenate((h,np.flipud(h[1:-1])))
        hflip = np.concatenate((h[0:1], np.flipud(h[1:])))
        return toeplitz(hflip,h)

    elif kernel.lower() == "vonmises":
        if kernel_param is None: kernel_param = 5
        h = np.exp(np.cos(2*np.pi*grid))
        h = (h/h[0])**kernel_param
        h = np.concatenate((h,np.flipud(h[1:-1])))
        hflip = np.concatenate((h[0:1], np.flipud(h[1:])))
        return toeplitz(hflip,h)

    else:
        raise NotImplementedError("This kernel is not implemented")

def _getExactSolution(dim,phantom,phantom_param):
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
                    'random' - random solution created according to the
                            prior specified in ???
            param = a parameter that determines the width of the central
                    "bump" of the function; the larger the parameter, the
                    narrower the "bump."  Does not apply to phantom = 'bumps'.
    
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
        x = np.zeros(dim)
        dimh = int(np.round(dim/2))
        w = int(np.round(dim/phantom_param))
        x[(dimh-w):(dimh+w)] = 1
        return x

    elif phantom.lower() == "hat":
        if phantom_param is None: phantom_param = 15
        x = np.zeros(dim)
        dimh = int(np.round(dim/2))
        w = int(np.round(dim/phantom_param))
        x[(dimh-w-1):(dimh)] = np.arange(w+1)/w
        x[(dimh-1):(dimh+w)] = np.flipud(np.arange(w+1))/w
        return x

    elif phantom.lower() == "bumps":
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

    else:
        raise NotImplementedError("This phantom is not implemented")
