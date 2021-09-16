import cuqi
import warnings
import numpy as np

try:
    import astra
except Exception as error:
    warnings.warn(error.msg)

class CT(cuqi.model.LinearModel):
    """Base CT class"""
    
    def __init__(self, proj_type="parallel", im_size=(45,45), det_count=50, angles=np.linspace(0,np.pi,60)):
        """TODO documentation of calls"""

        proj_geom = astra.create_proj_geom(proj_type,1.0,det_count,angles)
        vol_geom = astra.create_vol_geom(im_size[0],im_size[1])
        proj_id = astra.create_projector("linear",proj_geom,vol_geom)

        #TODO: refine geometries to reflect CT problem
        # e.g. vol_geom["option"]["WindowMinX"] etc.
        range_geometry  = cuqi.geometry.Continuous2D((np.size(angles),det_count))
        domain_geometry = cuqi.geometry.Continuous2D(im_size)

        forward = lambda x: astra.create_sino(np.reshape(x,im_size), proj_id)[1].ravel()
        adjoint = lambda y: astra.create_backprojection(np.reshape(y,(np.size(angles),det_count)),proj_id)[1].ravel()

        super().__init__(forward,adjoint,range_geometry,domain_geometry)

class CTshifted(cuqi.model.LinearModel):
    """CT model with source+detector shift"""

    def __init__(self,proj_type="fanflat_vec",
                      im_size=(45,45),
                      det_count=50,
                      angles=np.linspace(0,2*np.pi,60),
                      shift = -125.3, stc = 600, ctd = 500, dl = 411, domain=550):

        ds = dl/det_count

        #Define scan vectors
        s0 = np.array([shift, stc])
        d0 = np.array([shift, -ctd])
        u0 = np.array([ds, 0])
        vectors = np.empty([np.size(angles), 6])
        for i, val in enumerate(angles):
            R = np.array([[np.cos(val), -np.sin(val)], [np.sin(val), np.cos(val)]])
            s = R @ s0
            d = R @ d0
            u = R @ u0
            vectors[i, 0:2] = s
            vectors[i, 2:4] = d
            vectors[i, 4:6] = u

        #Define geometries
        proj_geom = astra.create_proj_geom(proj_type, det_count, vectors)
        vol_geom = astra.create_vol_geom(im_size[0],im_size[1],-domain/2,domain/2,-domain/2,domain/2)
        proj_id = astra.create_projector('line_fanflat',proj_geom,vol_geom)

        #TODO: refine geometries to reflect CT problem
        # e.g. vol_geom["option"]["WindowMinX"] etc.
        range_geometry  = cuqi.geometry.Continuous2D((np.size(angles),det_count))
        domain_geometry = cuqi.geometry.Continuous2D(im_size)

        forward = lambda x: astra.create_sino(np.reshape(x,im_size), proj_id)[1].ravel()
        adjoint = lambda y: astra.create_backprojection(np.reshape(y,(np.size(angles),det_count)),proj_id)[1].ravel()

        super().__init__(forward,adjoint,range_geometry,domain_geometry)
        