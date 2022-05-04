import numpy as np
import cuqi
from cil.plugins.tigre import ProjectionOperator
from cil.framework import ImageGeometry, AcquisitionGeometry, DataContainer

class CILModel(cuqi.model.LinearModel):
    """ Base class of cuqi model using CIL for CT projectors.

    Parameters
    -----------
    acquisition_geometry : CIL acquisition geometry.
        See CIL documentation.

    image_geometry : CIL image geometry.
        See CIL documentation.

    Attributes
    -----------
    range_geometry : cuqi.geometry.Image2D
        The geometry representing the range associated with sinogram.

    domain_geometry : cuqi.geometry.Image2D
        The geometry representing the domain associated with input image.

    ProjectionOperator : CIL ProjectionOperator
        The projection operator handling forwad and adjoint operations.

    Methods
    -----------
    :meth:`forward` the forward operator.
    :meth:`adjoint` the adjoint operator.

    """
    
    def __init__(self, acquisition_geometry: AcquisitionGeometry, image_geometry: ImageGeometry) -> None:

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
        self._fill_container_from_numpy(x, self._image_data)
        self.ProjectionOperator.direct(self._image_data, out=self._acquisition_data)
        return self._acquisition_data.as_array()

    def _adjoint_func(self, x: np.ndarray) -> np.ndarray:
        self._fill_container_from_numpy(x, self._acquisition_data)
        self.ProjectionOperator.adjoint(self._acquisition_data, out=self._image_data)
        return self._image_data.as_array()

    @staticmethod
    def _fill_container_from_numpy(array: np.ndarray, container: DataContainer):
        """ Fill a numpy array into a CIL data container without creating a copy. """

        # Check shape
        if array.shape != container.shape:
            raise ValueError("Array shape does not match container shape.")
            
        # Convert to dtype of container only if necessary (this is the main cost)
        if array.dtype != container.dtype: array = array.astype(container.dtype) 

        # Storing directly in .array avoids copying
        container.array = array

class ParallelBeam2DModel(CILModel):
    """ 2D CT model with parallel beam.

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

    """
    
    def __init__(self,
        im_size = (45,45),
        det_count = 50,
        angles = np.linspace(0,np.pi,60),
        det_spacing = None,
        domain = None
        ):

        if domain == None:
            domain = im_size

        if det_spacing is None:
            det_spacing = 1
            # TODO. Use default det_spacing that ensures whole domain is covered.
            # This allows changing im_size or det_count while keeping the same default scan area.
            # Current implementation with tigre causes some severe artifacts if we modify det_spacing.
            # det_spacing = np.sqrt(2) * np.max(domain) / det_count

        # Setup cil geometries for parallel beam CT
        acquisition_geometry = AcquisitionGeometry.create_Parallel2D()
        acquisition_geometry.set_angles(angles, angle_unit="radian")
        acquisition_geometry.set_panel(det_count, pixel_size=det_spacing)
        
        # Setup image geometry
        image_geometry = ImageGeometry(
            voxel_num_x=im_size[0],
            voxel_num_y=im_size[1],
            voxel_size_x=domain[0] / im_size[0],
            voxel_size_y=domain[1] / im_size[1],
        )

        super().__init__(acquisition_geometry, image_geometry)

class FanBeam2DModel(CILModel):
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

    """
    
    def __init__(self,
        im_size = (45,45),
        det_count = 50,
        angles = np.linspace(0,2*np.pi,60),
        source_object_dist = 200,
        object_detector_dist = 30,
        det_spacing = None,
        domain = None
        ):

        if domain == None:
            domain = im_size

        if det_spacing is None:
            det_spacing = 1

        # Setup cil geometries for parallel beam CT 
        acquisition_geometry = AcquisitionGeometry.create_Cone2D(
            source_position=[0.0, -source_object_dist],
            detector_position=[0.0, object_detector_dist],
        )
        acquisition_geometry.set_angles(angles, angle_unit="radian")
        acquisition_geometry.set_panel(det_count, pixel_size=det_spacing)

        # Setup image geometry
        image_geometry = ImageGeometry(
            voxel_num_x=im_size[0],
            voxel_num_y=im_size[1],
            voxel_size_x=domain[0] / im_size[0],
            voxel_size_y=domain[1] / im_size[1],
        )

        super().__init__(acquisition_geometry, image_geometry)

class ShiftedFanBeam2DModel(CILModel):
    """
    2D CT model with fanbeam and source+detector shift, assuming object is at position (0,0)

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

    det_spacing : int, default 1
        Detector element size/spacing.

    domain : tuple
        Size of image domain, default (550,550).

    """

    def __init__(self,
        im_size = (45,45),
        det_count = 50,
        angles = np.linspace(0,2*np.pi,60),
        source_y = -600,
        detector_y = 500,
        beamshift_x = -125.3,
        domain = (550,550),
        det_spacing = None
        ):

        if det_spacing is None:
            det_spacing = 1

        # Setup cil geometries for parallel beam CT with shifted source and detector
        acquisition_geometry = AcquisitionGeometry.create_Cone2D(
            source_position=[beamshift_x, source_y],
            detector_position=[beamshift_x, detector_y]
        )
        acquisition_geometry.set_angles(angles, angle_unit="radian")
        acquisition_geometry.set_panel(det_count, pixel_size=det_spacing)

        # Setup image geometry
        image_geometry = ImageGeometry(
            voxel_num_x=im_size[0],
            voxel_num_y=im_size[1],
            voxel_size_x=domain[0] / im_size[0],
            voxel_size_y=domain[1] / im_size[1],
        )

        super().__init__(acquisition_geometry, image_geometry) 

