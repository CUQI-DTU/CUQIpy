import cuqi
import numpy as np
import warnings
from scipy.interpolate import interp2d

try: 
    import astra #ASTRA Toolbox is used for all tomography projections
except Exception as error:
    warnings.warn(error.msg)


class _astraCT2D(cuqi.model.LinearModel):
    """ Base cuqi model using ASTRA for CT 2D projectors"""
    def __init__(self,
        beam_type,
        proj_type,
        im_size,
        det_count,
        det_spacing,
        angles=None,
        vectors=None,
        domain=None
        ):

        if angles is None and vectors is None:
            raise ValueError("Angles or vectors need to be specified")

        if angles is not None and vectors is not None:
            warnings.warn("Angles and vectors are both defined. Vectors will take prescedent.")

        # Default to square image size if scalar is given
        if not hasattr(im_size,"__len__"):
            im_size = (im_size,im_size)

        # Default to im_size domain size if none is given
        if domain is None:
            domain = im_size[0]
            
        # Set up astra projector
        if vectors is not None:
            proj_geom = astra.create_proj_geom(beam_type,det_count,vectors)
        else:
            proj_geom = astra.create_proj_geom(beam_type,det_spacing,det_count,angles)
        vol_geom = astra.create_vol_geom(im_size[0],im_size[1],-domain/2,domain/2,-domain/2,domain/2)
        proj_id = astra.create_projector(proj_type,proj_geom,vol_geom)

        # Domain geometry
        xgrid = np.linspace(vol_geom["option"]["WindowMinX"],vol_geom["option"]["WindowMaxX"],im_size[0])
        ygrid = np.linspace(vol_geom["option"]["WindowMinY"],vol_geom["option"]["WindowMaxY"],im_size[1])
        domain_geometry = cuqi.geometry.Continuous2D(grid=(xgrid,ygrid))

        # Range geometry
        if vectors is None:
            q = angles.shape[0]
            x_axis = np.rad2deg(angles)
        else:
            q = vectors.shape[0]
            x_axis = np.arange(q)
        range_geometry  = cuqi.geometry.Continuous2D(grid=(x_axis,np.arange(det_count)))
        
        super().__init__(self.forward,self.adjoint,range_geometry,domain_geometry)

        # Store other CT related variables privately
        self._proj_geom = proj_geom
        self._vol_geom = vol_geom
        self._proj_id = proj_id

    # Getter methods for private variables
    @property
    def proj_geom(self):
        return self._proj_geom
    @property
    def vol_geom(self):
        return self._vol_geom
    @property
    def proj_id(self):
        return self._proj_id

    # CT forward projection
    def forward(self,x):
        id, sinogram =  astra.create_sino(np.reshape(x,self.domain_geometry.shape,order='F'), self.proj_id)
        astra.data2d.delete(id)
        return sinogram.flatten(order='F')

    # CT back projection
    def adjoint(self,y):
        id, volume = astra.create_backprojection(np.reshape(y,self.range_geometry.shape,order='F'),self.proj_id)
        astra.data2d.delete(id)
        return volume.flatten(order='F')


class CT2D_basic(_astraCT2D):
    """2D CT model defined by the angle of the scan"""
    
    def __init__(self,
        beam_type="parallel",
        proj_type = "linear",
        im_size=(45,45),
        det_count=50,
        det_spacing=1,
        angles=np.linspace(0,np.pi,60)
        ):
        """Initialize base CT"""

        super().__init__(beam_type,proj_type,im_size,det_count,det_spacing,angles)

class CT2D_shifted(_astraCT2D):
    """2D CT model with source+detector shift"""

    def __init__(self,beam_type="fanflat_vec",
                      proj_type='line_fanflat',
                      im_size=(45,45),
                      det_count=50,
                      angles=np.linspace(0,2*np.pi,60),
                      beamshift_x = -125.3, source_y = -600, detector_y = 500, dl = 411, domain=550):
        
        # Detector spacing
        det_spacing = dl/det_count

        #Define scan vectors
        s0 = np.array([beamshift_x, source_y])
        d0 = np.array([beamshift_x , detector_y])
        u0 = np.array([det_spacing, 0])
        vectors = np.empty([np.size(angles), 6])
        for i, val in enumerate(angles):
            R = np.array([[np.cos(val), -np.sin(val)], [np.sin(val), np.cos(val)]])
            s = R @ s0
            d = R @ d0
            u = R @ u0
            vectors[i, 0:2] = s
            vectors[i, 2:4] = d
            vectors[i, 4:6] = u

        super().__init__(beam_type,proj_type,im_size,det_count,det_spacing,None,vectors,domain)        

#=============================================================================
class ParBeamCT_2D(cuqi.problem.BayesianProblem):
    """
    2D parallel-beam Computed Tomography test problem using ASTRA

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
        ):

        #######
        model = CT2D_basic() #CT model with default values
        #model= CT2D_shifted() #Shifted detector/source CT model with default values

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
        x_exact = x_exact.ravel()
        x_exact = cuqi.samples.CUQIarray(x_exact, is_par=True, geometry=model.domain_geometry)

        #model.domain_geometry.plot(x_exact); plt.title("Phantom")
        # Generate exact data
        b_exact = model.forward(x_exact)

        # Define and add noise #TODO: Add Poisson and logpoisson
        if noise_type.lower() == "gaussian":
            likelihood = cuqi.distribution.Gaussian(model,noise_std,np.eye(dim))
        elif noise_type.lower() == "scaledgaussian":
            likelihood = cuqi.distribution.Gaussian(model,b_exact*noise_std,np.eye(dim))
        else:
            raise NotImplementedError("This noise type is not implemented")
        
        # Generate data
        if data is None:
            data = likelihood(x=x_exact).sample() #ToDo: (remove flatten)

        # Initialize Deconvolution as BayesianProblem problem
        super().__init__(likelihood,prior,data)

        # Store exact values
        self.exactSolution = x_exact
        self.exactData = b_exact
        self.infoString = "Noise type: Additive {} with std: {}".format(noise_type.capitalize(),noise_std)

        """
        # Set up model
        A = _getCirculantMatrix(dim,kernel,kernel_param)
        model = cuqi.model.LinearModel(A,range_geometry=Continuous1D(dim),domain_geometry=Continuous1D(dim))

        # Set up exact solution
        x_exact = _getExactSolution(dim,phantom,phantom_param)
        x_exact = CUQIarray(x_exact, geometry=model.domain_geometry)

        # Generate exact data
        b_exact = model.forward(x_exact)

        # Define and add noise #TODO: Add Poisson and logpoisson
        if noise_type.lower() == "gaussian":
            likelihood = cuqi.distribution.Gaussian(model,noise_std,np.eye(dim))
        elif noise_type.lower() == "scaledgaussian":
            likelihood = cuqi.distribution.Gaussian(model,b_exact*noise_std,np.eye(dim))
        else:
            raise NotImplementedError("This noise type is not implemented")
        
        # Generate data
        if data is None:
            data = likelihood(x=x_exact).sample() #ToDo: (remove flatten)

        # Initialize Deconvolution as BayesianProblem problem
        super().__init__(likelihood,prior,data)

        # Store exact values
        self.exactSolution = x_exact
        self.exactData = b_exact
        self.infoString = "Noise type: Additive {} with std: {}".format(noise_type.capitalize(),noise_std)
        """
