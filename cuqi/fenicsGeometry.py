from cuqi.geometry import Geometry
import numpy as np
import dolfin as dl
import matplotlib.pyplot as plt
import math


class FenicsContinuous(Geometry):

    def __init__(self, Vh, labels = ['x', 'y']):
        self._physical_dim = Vh.mesh().geometry().dim()
        if self.physical_dim >2:
            raise NotImplementedError("'FenicsContinuous' object does not support 3D meshes yet. 'mesh' needs to be a 1D or 2D mesh.")
        self.labels = labels
        self.Vh = Vh

    @property
    def physical_dim(self):
        return self._physical_dim   

    @property
    def mesh(self):
        return self.Vh.mesh()

    def plot(self,values,subplots=True,**kwargs):
        """
        Overrides :meth:`cuqi.geometry.Geometry.plot`. See :meth:`cuqi.geometry.Geometry.plot` for description  and definition of the parameter `values`.
        
        Parameters
        -----------
        kwargs : keyword arguments
            keyword arguments which the function :meth:`dolfin.plot` normally takes.
        """
        values = self._process_values(values)
        subplot_ids = self._create_subplot_list(values,subplots=subplots)

        ims = []
        func = dl.Function(self.Vh)
        for rows,cols,id in subplot_ids:
            func.vector().zero()
            func.vector().set_local(values[...,id-1])
            if subplots:
                plt.subplot(rows,cols,id); 
            ims.append(dl.plot(func, **kwargs))

        self._plot_config(subplots) 
        return ims

    def _process_values(self, values):
        if len(values.shape) == 1:
            values = values[..., np.newaxis]
        
        return values
    
    def _plot_config(self, subplot):
        if self.labels is not None:
            if subplot == False:
                plt.gca().set_xlabel(self.labels[0])
                if self.physical_dim == 2: plt.gca().set_ylabel(self.labels[1]) 
            else:
                for i, axis in enumerate(plt.gcf().axes):
                    axis.set_xlabel(self.labels[0])
                    if self.physical_dim == 2: axis.set_ylabel(self.labels[1])
    