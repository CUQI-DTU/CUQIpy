import cuqi
from cil.plugins.tigre import ProjectionOperator
from cil.framework import ImageData
from cil.framework import AcquisitionData
from .geometry import cilAcquisitionGeometry, cilImageGeometry


class cilBase(cuqi.model.LinearModel):
    
    def __init__(self, ig, ag) -> None:
        self.ig = ig
        self.ag = ag
        self.range_geometry = cilAcquisitionGeometry(self.ag)
        self.domain_geometry = cilImageGeometry(self.ig)
        super().__init__(self.forward, self.adjoint, domain_geometry = self.domain_geometry, range_geometry = self.range_geometry)

        # Create projection operator using Tigre.
        self.ProjOp = ProjectionOperator(ig, ag)

    def forward(self,x):
        x_cil = ImageData(array = x.reshape(self.domain_geometry.shape).__array__(), geometry = self.ig) # Maybe move to init
        proj_cil = self.ProjOp.direct(x_cil)
        out = cuqi.samples.CUQIarray(proj_cil.as_array().flatten(), geometry=self.range_geometry, is_par = False)
        return out

    def adjoint(self,x):
        x_cil = AcquisitionData(array = x.reshape(self.range_geometry.shape).__array__(), geometry = self.ag)
        adj_cil = self.ProjOp.adjoint(x_cil)
        out = cuqi.samples.CUQIarray(adj_cil.as_array().flatten(), geometry=self.domain_geometry, is_par = False)
        return out


