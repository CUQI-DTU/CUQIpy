from cuqi.geometry import Geometry
from cil.utilities.display import show2D
from cil.framework import ImageData
from cil.framework import AcquisitionData

from cuqi.samples import CUQIarray

class cilGeometry(Geometry):

    def __init__(self,cil_data):

        self.cil_data = cil_data
        self.is_par = True
    
    @property
    def shape(self):
        return self.cil_data.shape

    def par2fun(self, pars):
        out = pars.reshape(self.shape+(-1,)).squeeze().astype('float32')
        return out

    def fun2par(self, funvals):
        return funvals.ravel() #Maybe use reshape((self.dim,), order=self.order)
    
    def _plot(self,values, **kwargs):
        # Visualise data
        self.cil_data.fill(values)
        show2D(self.cil_data, **kwargs)#, 'simulated sinogram', cmap=cmap, size=(10,10), origin='upper-left')