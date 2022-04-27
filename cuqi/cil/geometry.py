from cuqi.geometry import Geometry
from cil.utilities.display import show2D
from cil.framework import ImageData
from cil.framework import AcquisitionData

from cuqi.samples import CUQIarray
import matplotlib.pyplot as plt

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
        # Visualise data with cil's plotting functionality

        values = self._process_values(values)
        subplot_ids = self._create_subplot_list(values.shape[-1])

        plots = []
        if len(subplot_ids)==1:
            num_cols = 1
        elif len(subplot_ids)==2:
            num_cols = 2
        else:
            num_cols = 3

        for rows, cols, subplot_id in subplot_ids:
            self.cil_data.fill(values[...,subplot_id-1])
            plots.append(self.cil_data)
        ims = show2D(plots, num_cols = num_cols, **kwargs)
        return ims

    def _process_values(self,values):
        if len(values.shape) == 3 or\
             (len(values.shape) == 2 and values.shape[0]== self.dim):  
            pass
        else:
            values = values[..., np.newaxis]
        return values