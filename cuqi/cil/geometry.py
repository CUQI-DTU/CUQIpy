from cuqi.geometry import Geometry
from cil.utilities.display import show2D
from cil.framework import ImageData
from cil.framework import AcquisitionData

from cuqi.samples import CUQIarray

import numpy as np

class cilGeometry(Geometry):
    """ A geometry used with cil models.

    The par2fun method converts the parameter vector into an image (matrix) and ensures it is float32 required in cil.
    The fun2par method converts the image (matrix) into a parameter vector.

    Plotting is handled via cil's plotting functionality.

    Parameters
    -----------
    cil_data : cil DataContainer
        See cil documentation
    
    Attributes
    -----------
    shape : tuple
        shape of the image (rows, columns)
    """

    def __init__(self,cil_data):

        self.cil_data = cil_data
        self.is_par = True

    def __eq__(self, other):
        return np.all(self.cil_data == other.cil_data)

    @property
    def shape(self):
        return self.cil_data.shape

    def par2fun(self, pars):
        out = pars.reshape(self.shape+(-1,)).squeeze().astype('float32')
        return out

    def fun2par(self, funvals):
        return funvals.ravel() 
    
    def _plot(self,values, **kwargs):
        # Visualise data
        self.cil_data.fill(values)
        show2D(self.cil_data, **kwargs)