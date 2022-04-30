import cuqi
import numpy as np
import warnings
from scipy.interpolate import interp2d
import scipy.io as io

import astra


class astraBase2D(cuqi.model.LinearModel): # 2D because of DetectorCount and Image2D geometries
    """ Base cuqi model using ASTRA for CT projectors.

    Parameters
    -----------
    proj_type : string.
        String indication astra projection type. Could be "linear", "cuda" ect. See astra documentation. 

    proj_geom : astra projection geometry.
        See astra documentation.

    vol_geom : astra volume geometry.
        See astra documentation.

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
    def __init__(self, proj_type, proj_geom, vol_geom):

        # Astra projection id
        proj_id = astra.create_projector(proj_type,proj_geom,vol_geom)

        # Domain geometry
        #xgrid = np.linspace(vol_geom["option"]["WindowMinX"],vol_geom["option"]["WindowMaxX"],vol_geom["GridRowCount"])
        #ygrid = np.linspace(vol_geom["option"]["WindowMinY"],vol_geom["option"]["WindowMaxY"],vol_geom["GridColCount"])
        domain_geometry = cuqi.geometry.Image2D(shape = (vol_geom["GridRowCount"],vol_geom["GridColCount"]), order = "F")
       
        # Range geometry
        if "Vectors" in proj_geom:
            num_angles = proj_geom["Vectors"].shape[0]
        else:
            num_angles = proj_geom["ProjectionAngles"].shape[0]
        range_geometry = cuqi.geometry.Image2D(shape = (num_angles, proj_geom["DetectorCount"]), order = "F")
        
        super().__init__(self._forward_func,self._adjoint_func,range_geometry,domain_geometry)

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
    def _forward_func(self,x):
        id, sinogram =  astra.create_sino(x, self.proj_id)
        astra.data2d.delete(id)
        return sinogram

    # CT back projection
    def _adjoint_func(self,y):
        id, volume = astra.create_backprojection(y,self.proj_id)
        astra.data2d.delete(id)
        return volume


class CT2D_parallel(astraBase2D):
    """2D CT model with parallel beam"""
    
    def __init__(self,
        proj_type = "linear",
        im_size=(45,45),
        det_count=50,
        det_spacing=1,
        angles=np.linspace(0,np.pi,60),
        domain = None
        ):

        if domain == None:
            domain = im_size

        """Setup astra geometries for parallel beam CT"""
        proj_geom = astra.create_proj_geom("parallel",det_spacing,det_count,angles)
        vol_geom = astra.create_vol_geom(im_size[0],im_size[1],-domain[0]/2,domain[0]/2,-domain[1]/2,domain[1]/2)

        super().__init__(proj_type, proj_geom, vol_geom)

class CT2D_fanbeam(astraBase2D):
    """2D CT model with parallel beam"""
    
    def __init__(self,
        proj_type = 'line_fanflat',
        im_size=(45,45),
        det_count=50,
        det_spacing=1,
        angles=np.linspace(0,np.pi,60),
        source_object_dist = 200,
        object_detector_dist = 30,
        domain = None
        ):

        if domain == None:
            domain = im_size

        """Setup astra geometries for parallel beam CT"""
        proj_geom = astra.create_proj_geom("fanflat",det_spacing,det_count,angles, source_object_dist, object_detector_dist)
        vol_geom = astra.create_vol_geom(im_size[0],im_size[1],-domain[0]/2,domain[0]/2,-domain[1]/2,domain[1]/2)

        super().__init__(proj_type, proj_geom, vol_geom)

class CT2D_shiftedfanbeam(astraBase2D):
    """2D CT model with source+detector shift"""

    def __init__(self,beam_type="fanflat_vec",
                      proj_type='line_fanflat',
                      im_size=(45,45),
                      det_count=50,
                      angles=np.linspace(0,2*np.pi,60),
                      beamshift_x = -125.3, source_y = -600, detector_y = 500, dl = 411, domain=(550,550)):
        
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

        # Astra geometries
        proj_geom = astra.create_proj_geom(beam_type,det_count,vectors)
        vol_geom = astra.create_vol_geom(im_size[0],im_size[1],-domain[0]/2,domain[0]/2,-domain[1]/2,domain[1]/2)

        super().__init__(proj_type, proj_geom, vol_geom)        



