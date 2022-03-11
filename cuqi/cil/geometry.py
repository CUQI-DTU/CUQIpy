from cuqi.geometry import Geometry
from cil.utilities.display import show2D
from cil.framework import ImageData
from cil.framework import AcquisitionData

class cilAcquisitionGeometry(Geometry):

    def __init__(self,cilag):

        self.cilag = cilag
    
    @property
    def shape(self):
        return self.cilag.shape
    
    def _plot(self,values):
        # Visualise data
        values_cil = AcquisitionData(array = values.reshape(self.cilag.shape).__array__(), geometry = self.cilag)
        show2D(values_cil)#, 'simulated sinogram', cmap=cmap, size=(10,10), origin='upper-left')

class cilImageGeometry(Geometry):

    def __init__(self,cilig):

        self.cilig = cilig
    
    @property
    def shape(self):
        return self.cilig.shape
    
    def _plot(self,values):
        # Visualise data
        values_cil = ImageData(array = values.reshape(self.cilig.shape).__array__(), geometry = self.cilig)
        show2D(values_cil)#, 'simulated sinogram', cmap=cmap, size=(10,10), origin='upper-left')
