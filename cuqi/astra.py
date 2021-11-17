import cuqi
import numpy as np
import warnings
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