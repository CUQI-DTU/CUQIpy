import cuqi
from cil.plugins.tigre import ProjectionOperator
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

