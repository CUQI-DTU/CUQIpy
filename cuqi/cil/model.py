import numpy as np
import cuqi
from cil.plugins.tigre import ProjectionOperator
from cil.framework import ImageGeometry, AcquisitionGeometry
from .geometry import cilGeometry


class cilBase(cuqi.model.LinearModel):
    """ Base cuqi model using CIL for CT projectors.

    Parameters
    -----------
    acqu_geom : CIL acquisition geometry.
        See CIL documentation.

    im_geom : CIL image geometry.
        See CIL documentation.

    Attributes
    -----------
    range_geometry : cuqi.geometry.Geometry
        The geometry representing the range associated with sinogram.

    domain_geometry : cuqi.geometry.Geometry
        The geometry representing the domain associated with reconstructed image.

    Methods
    -----------
    :meth:`forward` the forward operator.
    :meth:`adjoint` the adjoint operator.

    """
    
    def __init__(self, acqu_geom, im_geom) -> None:
        self._acqu_geom = acqu_geom
        self._im_geom = im_geom
        self.acqu_data = acqu_geom.allocate()
        self.im_data = im_geom.allocate()
        self.range_geometry = cilGeometry(self.acqu_data)
        self.domain_geometry = cilGeometry(self.im_data)
        super().__init__(self._forward_func, self._adjoint_func, domain_geometry = self.domain_geometry, range_geometry = self.range_geometry)

        # Create projection operator using Tigre.
        self._ProjOp = ProjectionOperator(im_geom, acqu_geom)

    # Getter methods for private variables
    @property
    def acqu_geom(self):
        return self._acqu_geom
    @property
    def im_geom(self):
        return self._im_geom
    @property
    def ProjOp(self):
        return self._ProjOp

    def _forward_func(self,x):
        self.im_data.fill(x)
        proj_cil = self.ProjOp.direct(self.im_data)
        out = cuqi.samples.CUQIarray(proj_cil.as_array(), geometry = self.range_geometry, is_par = True)
        return out

    def _adjoint_func(self,x):
        self.acqu_data.fill(x)
        adj_cil = self.ProjOp.adjoint(self.acqu_data)
        out = cuqi.samples.CUQIarray(adj_cil.as_array(), geometry=self.domain_geometry, is_par = True)
        return out

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
        acqu_geom = AcquisitionGeometry.create_Parallel2D()\
                            .set_angles(angles, angle_unit ='radian')\
                            .set_panel(det_count, pixel_size=det_spacing)
        
        # Setup image geometry
        im_geom = ImageGeometry(voxel_num_x=im_size[0], 
                        voxel_num_y=im_size[1], 
                        voxel_size_x=domain[0]/im_size[0], 
                        voxel_size_y=domain[1]/im_size[1])

        super().__init__(acqu_geom, im_geom)

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
        acqu_geom = AcquisitionGeometry.create_Cone2D(\
            source_position=[0.0, -source_object_dist],\
            detector_position=[0.0, object_detector_dist])
        acqu_geom.set_angles(angles, angle_unit ='radian')
        acqu_geom.set_panel(det_count, pixel_size=det_spacing)

        # Setup image geometry
        im_geom = ImageGeometry(voxel_num_x=im_size[0], 
                        voxel_num_y=im_size[1], 
                        voxel_size_x=domain[0]/im_size[0], 
                        voxel_size_y=domain[1]/im_size[1])

        super().__init__(acqu_geom, im_geom)

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
        acqu_geom = AcquisitionGeometry.create_Cone2D(\
            source_position = [beamshift_x, source_y],\
            detector_position = [beamshift_x, detector_y])
        acqu_geom.set_angles(angles, angle_unit ='radian')
        acqu_geom.set_panel(det_count, pixel_size=det_spacing)

        # Setup image geometry
        im_geom = ImageGeometry(voxel_num_x=im_size[0], 
                        voxel_num_y=im_size[1], 
                        voxel_size_x=domain[0]/im_size[0], 
                        voxel_size_y=domain[1]/im_size[1])

        super().__init__(acqu_geom, im_geom) 

