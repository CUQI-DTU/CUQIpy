import numpy as np
import cuqi
from cil.plugins.tigre import ProjectionOperator
from cil.framework import ImageGeometry, AcquisitionGeometry, DataContainer

class cilBase(cuqi.model.LinearModel):
    """ Base cuqi model using CIL for CT projectors.

    Parameters
    -----------
    acquisition_geometry : CIL acquisition geometry.
        See CIL documentation.

    image_geometry : CIL image geometry.
        See CIL documentation.

    Attributes
    -----------
    range_geometry : cuqi.geometry.Geometry
        The geometry representing the range associated with sinogram.

    domain_geometry : cuqi.geometry.Geometry
        The geometry representing the domain associated with reconstructed image.

    ProjectionOperator : CIL ProjectionOperator
        The projection operator handling forwad and adjoint operations.

    Methods
    -----------
    :meth:`forward` the forward operator.
    :meth:`adjoint` the adjoint operator.

    """
    
    def __init__(self, acquisition_geometry : AcquisitionGeometry, image_geometry : ImageGeometry) -> None:

        # Define image geometries
        range_geometry = cuqi.geometry.Image2D(acquisition_geometry.shape)
        domain_geometry = cuqi.geometry.Image2D(image_geometry.shape)
        super().__init__(self._forward_func, self._adjoint_func, domain_geometry=domain_geometry, range_geometry=range_geometry)

        # Create projection operator using Tigre.
        self._ProjectionOperator = ProjectionOperator(image_geometry, acquisition_geometry)
        
        # Allocate data containers for efficiency
        self._acquisition_data = acquisition_geometry.allocate()
        self._image_data = image_geometry.allocate()

    
    @property
    def acquisition_geometry(self):
        """ The CIL acquisition geometry. """
        return self.ProjectionOperator.range_geometry()

    @property
    def image_geometry(self):
        """ The CIL image geometry. """
        return self.ProjectionOperator.domain_geometry()

    @property
    def ProjectionOperator(self):
        """ The CIL projection operator. """
        return self._ProjectionOperator

    def _forward_func(self, x: np.ndarray) -> np.ndarray:
        self._fill_from_numpy(x, self._image_data)
        self.ProjectionOperator.direct(self._image_data, out=self._acquisition_data)
        return self._acquisition_data.as_array()

    def _adjoint_func(self, x: np.ndarray) -> np.ndarray:
        self._fill_from_numpy(x, self._acquisition_data)
        self.ProjectionOperator.adjoint(self._acquisition_data, out=self._image_data)
        return self._image_data.as_array()

    def _fill_from_numpy(self, x: np.ndarray, data_container: DataContainer):
        """ Fill a numpy array into a CIL data container without creating a copy. """
        # Convert to dtype of container only if necessary (this is the main potential cost)
        if x.dtype != np.float32: x = x.astype(data_container.dtype) 

        # Storing directly in .array avoids copying
        data_container.array = x 

class CT2D_parallel(cilBase):
    """
    2D CT model with parallel beam

    Parameters
    ------------    
    im_size : tuple
        Dimensions of image in pixels, default (45,45).
    
    det_count : int
        Number of detector elements, default 50.
    
    det_spacing : int
        detector element size/spacing, default 1.
    
    angles : ndarray
        Angles of projections, in radians, 
        default np.linspace(0,np.pi,60).

    domain : tuple
        Size of image domain, default domain = im_size

    """
    
    def __init__(self,
        im_size = (45,45),
        det_count = 50,
        det_spacing = 1,
        angles = np.linspace(0,np.pi,60),
        domain = None
        ):

        if domain == None:
            domain = im_size

        # Setup cil geometries for parallel beam CT
        acquisition_geometry = AcquisitionGeometry.create_Parallel2D()\
                            .set_angles(angles, angle_unit ='radian')\
                            .set_panel(det_count, pixel_size=det_spacing)
        
        # Setup image geometry
        image_geometry = ImageGeometry(voxel_num_x=im_size[0], 
                        voxel_num_y=im_size[1], 
                        voxel_size_x=domain[0]/im_size[0], 
                        voxel_size_y=domain[1]/im_size[1])

        super().__init__(acquisition_geometry, image_geometry)

class CT2D_fanbeam(cilBase):
    """
    2D CT model with fan beam, assuming centered beam

    Parameters
    ------------    
    im_size : tuple
        Dimensions of image in pixels, default (45,45).
    
    det_count : int
        Number of detector elements, default 50.
    
    det_spacing : int
        detector element size/spacing, default 1.
    
    angles : ndarray
        Angles of projections, in radians, 
        default np.linspace(0,np.pi,60).

    source_object_dist : scalar
        Distance between source and object, default 200.

    object_detector_dist : scalar
        Distance between detector and object, default 30.

    domain : tuple
        Size of image domain, default domain = im_size

    """
    
    def __init__(self,
        im_size = (45,45),
        det_count = 50,
        det_spacing = 1,
        angles = np.linspace(0,np.pi,60),
        source_object_dist = 200,
        object_detector_dist = 30,
        domain = None
        ):

        if domain == None:
            domain = im_size

        # Setup cil geometries for parallel beam CT 
        acquisition_geometry = AcquisitionGeometry.create_Cone2D(\
            source_position=[0.0, -source_object_dist],\
            detector_position=[0.0, object_detector_dist])
        acquisition_geometry.set_angles(angles, angle_unit ='radian')
        acquisition_geometry.set_panel(det_count, pixel_size=det_spacing)

        # Setup image geometry
        image_geometry = ImageGeometry(voxel_num_x=im_size[0], 
                        voxel_num_y=im_size[1], 
                        voxel_size_x=domain[0]/im_size[0], 
                        voxel_size_y=domain[1]/im_size[1])

        super().__init__(acquisition_geometry, image_geometry)

class CT2D_shiftedfanbeam(cilBase):
    """
    2D CT model with fanbeam and source+detector shift, assuming object is at position (0,0)

    Parameters
    ------------    
    im_size : tuple
        Dimensions of image in pixels, default (45,45).
    
    det_count : int
        Number of detector elements, default 50.
    
    det_spacing : int
        detector element size/spacing, default 1.
    
    angles : ndarray
        Angles of projections, in radians, 
        default np.linspace(0,np.pi,60).

    source_y : scalar
        Source position on y-axis, default -600.

    detector_y : scalar
        Detector position on y-axis, default 500.

    beamshift_x : scalar
        Source and detector position on x-axis, default -125.3.

    domain : tuple
        Size of image domain, default (550,550).

    """

    def __init__(self,
        im_size = (45,45),
        det_count = 50,
        det_spacing = 1,
        angles = np.linspace(0,2*np.pi,60),
        source_y = -600,
        detector_y = 500,
        beamshift_x = -125.3,
        domain = (550,550)):

        # Setup cil geometries for parallel beam CT with shifted source and detector
        acquisition_geometry = AcquisitionGeometry.create_Cone2D(\
            source_position = [beamshift_x, source_y],\
            detector_position = [beamshift_x, detector_y])
        acquisition_geometry.set_angles(angles, angle_unit ='radian')
        acquisition_geometry.set_panel(det_count, pixel_size=det_spacing)

        # Setup image geometry
        image_geometry = ImageGeometry(voxel_num_x=im_size[0], 
                        voxel_num_y=im_size[1], 
                        voxel_size_x=domain[0]/im_size[0], 
                        voxel_size_y=domain[1]/im_size[1])

        super().__init__(acquisition_geometry, image_geometry) 

