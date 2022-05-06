import cuqi
import numpy as np
import astra

class ASTRAModel(cuqi.model.LinearModel):
    """ Base class of cuqi model using ASTRA Toolbox for CT projectors.

    Currently only supports 2D CT.

    For more details on the projectors, see:
    https://www.astra-toolbox.com

    Parameters
    -----------
    proj_type : string
        String indication projection type. 
        Could be "line", "strip", "linear", 
        "cuda", "line_fanflat", "strip_fanflat" etc.

    proj_geom : dict
        ASTRA projection geometry.

    vol_geom : dict
        ASTRA volume geometry.

    Attributes
    -----------
    range_geometry : cuqi.geometry.Geometry
        The geometry representing the range associated with sinogram.

    domain_geometry : cuqi.geometry.Geometry
        The geometry representing the domain associated with input image.

    proj_id : int
        The ID of the ASTRA projector handling the forward and adjoint operations.

    Methods
    -----------
    :meth:`forward` the forward operator.
    :meth:`adjoint` the adjoint operator.

    """
    def __init__(self, proj_type, proj_geom, vol_geom):


        # Define image (domain) geometry
        domain_geometry = cuqi.geometry.Image2D(shape=(vol_geom["GridRowCount"], vol_geom["GridColCount"]), order = "F")

        # Define sinogram (range) geometry
        num_angles = proj_geom["Vectors"].shape[0] if "Vectors" in proj_geom else proj_geom["ProjectionAngles"].shape[0]
        range_geometry = cuqi.geometry.Image2D(shape=(num_angles, proj_geom["DetectorCount"]), order = "F")
        
        # Define linear model
        super().__init__(self._forward_func, self._adjoint_func, range_geometry=range_geometry, domain_geometry=domain_geometry)

        # Create ASTRA projector
        self._proj_id = astra.create_projector(proj_type, proj_geom, vol_geom)

        # Store other ASTRA related variables privately
        self._proj_geom = proj_geom
        self._vol_geom = vol_geom

    @property
    def proj_geom(self):
        """ ASTRA projection geometry. """
        return self._proj_geom

    @property
    def vol_geom(self):
        """ ASTRA volume geometry. """
        return self._vol_geom

    @property
    def proj_id(self):
        """ ASTRA projector ID. """
        return self._proj_id

    # CT forward projection
    def _forward_func(self, x: np.ndarray) -> np.ndarray:
        id, sinogram =  astra.create_sino(x, self.proj_id)
        astra.data2d.delete(id)
        return sinogram

    # CT back projection
    def _adjoint_func(self, y: np.ndarray) -> np.ndarray:
        id, volume = astra.create_backprojection(y, self.proj_id)
        astra.data2d.delete(id)
        return volume

class ParallelBeam2DModel(ASTRAModel):
    """2D CT model with parallel beam
    
    Parameters
    ----------

    im_size : tuple of ints
        Dimensions of image in pixels.
    
    det_count : int
        Number of detector elements.
       
    angles : ndarray
        Angles of projections, in radians.

    det_spacing : float, default 1
        Detector element size/spacing.

    domain : tuple, default im_size
        Size of image domain.

    proj_type : string
        String indication projection type.
        Can be "line", "strip", "linear", "cuda" etc.

    """
    
    def __init__(self,
        im_size=(45,45),
        det_count=50,
        angles=np.linspace(0, np.pi, 60),
        det_spacing=None,
        domain = None,
        proj_type="linear",
        ):

        if domain == None:
            domain = im_size

        if det_spacing is None:
            det_spacing = 1

        proj_geom = astra.create_proj_geom("parallel", det_spacing, det_count, angles)
        vol_geom = astra.create_vol_geom(im_size[0], im_size[1], -domain[0]/2, domain[0]/2, -domain[1]/2, domain[1]/2)

        super().__init__(proj_type, proj_geom, vol_geom)

class FanBeam2DModel(ASTRAModel):
    """ 2D CT model with fan beam.
    
    Assumes a centered beam.

    Parameters
    ------------    
    im_size : tuple of ints
        Dimensions of image in pixels.
    
    det_count : int
        Number of detector elements.
    
    angles : ndarray
        Angles of projections, in radians.

    source_object_dist : scalar
        Distance between source and object.

    object_detector_dist : scalar
        Distance between detector and object.

    det_spacing : int, default 1
        Detector element size/spacing.

    domain : tuple, default im_size
        Size of image domain.

    proj_type : string
        String indication projection type.
        Can be "line_fanflat", "strip_fanflat", "cuda" etc.

    """
    
    def __init__(self,
        im_size=(45,45),
        det_count=50,
        angles=np.linspace(0, 2*np.pi, 60),
        source_object_dist=200,
        object_detector_dist=30,
        det_spacing=None,
        domain=None,
        proj_type='line_fanflat'
        ):

        if domain == None:
            domain = im_size

        if det_spacing is None:
            det_spacing = 1

        proj_geom = astra.create_proj_geom("fanflat", det_spacing, det_count, angles, source_object_dist, object_detector_dist)
        vol_geom = astra.create_vol_geom(im_size[0], im_size[1], -domain[0]/2, domain[0]/2, -domain[1]/2, domain[1]/2)

        super().__init__(proj_type, proj_geom, vol_geom)

class ShiftedFanBeam2DModel(ASTRAModel):
    """ 2D CT model with fanbeam and source+detector shift, assuming object is at position (0,0).

    Parameters
    ------------    
    im_size : tuple of ints
        Dimensions of image in pixels.
    
    det_count : int
        Number of detector elements.
       
    angles : ndarray
        Angles of projections, in radians.

    source_y : scalar
        Source position on y-axis.

    detector_y : scalar
        Detector position on y-axis.

    beamshift_x : scalar
        Source and detector position on x-axis.

    det_length : scalar
        Detector length.

    domain : tuple
        Size of image domain, default (550,550).

    proj_type : string
        String indication projection type.
        Can be "line_fanflat", "strip_fanflat", "cuda" etc.
    
    beam_type : string
        String indication beam type.
        Must be of _vec type.
        For example "fanflat_vec".
    """

    def __init__(
        self,
        im_size=(45,45),
        det_count=50,
        angles=np.linspace(0, 2*np.pi, 60),
        source_y=-600,
        detector_y=500,
        beamshift_x=-125.3,
        det_length=411,
        domain=(550,550),
        proj_type='line_fanflat',
        beam_type="fanflat_vec"
        ):
        
        # Detector spacing
        det_spacing = det_length/det_count

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

        # Astra geometries
        proj_geom = astra.create_proj_geom(beam_type, det_count, vectors)
        vol_geom = astra.create_vol_geom(im_size[0], im_size[1], -domain[0]/2, domain[0]/2, -domain[1]/2, domain[1]/2)

        super().__init__(proj_type, proj_geom, vol_geom)        



