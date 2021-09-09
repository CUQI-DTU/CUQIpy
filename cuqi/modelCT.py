import cuqi
import warnings
import numpy as np

try:
    import astra
except Exception as error:
    warnings.warn(error.msg)

class CT(cuqi.model.LinearModel):
    """Base CT class"""
    
    def __init__(self, proj_type="parallel", im_size=(128,128), det_count=128, angles=np.linspace(0,np.pi,128)):
        """TODO documentation of calls"""

        proj_geom = astra.create_proj_geom(proj_type,1.0,det_count,angles)
        vol_geom = astra.create_vol_geom(im_size[0],im_size[1])
        proj_id = astra.create_projector("linear",proj_geom,vol_geom)

        #TODO: refine geometries to reflect CT problem
        # e.g. vol_geom["option"]["WindowMinX"] etc.
        range_geometry  = cuqi.geometry.Continuous2D((det_count,np.size(angles)))
        domain_geometry = cuqi.geometry.Continuous2D(im_size)

        forward = lambda x: astra.create_sino(np.reshape(x,im_size), proj_id)[1].ravel()
        adjoint = lambda y: astra.create_backprojection(np.reshape(y,(det_count,np.size(angles))),proj_id)[1].ravel()

        super().__init__(forward,adjoint,range_geometry,domain_geometry)


