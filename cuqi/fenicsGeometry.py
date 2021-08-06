from cuqi.geometry import Geometry
import numpy as np
import dolfin as dl
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt


class FenicsContinuous(Geometry,ABC):

    @abstractmethod
    def __init__(self, Vh):
        self.Vh = Vh

    @property
    def mesh(self):
        return self.Vh.mesh()

    def plot(self,values,**kwargs):
        """
        Overrides :meth:`cuqi.geometry.Geometry.plot`. See :meth:`cuqi.geometry.Geometry.plot` for description  and definition of the parameter `values`.
        
        Parameters
        -----------
        kwargs : keyword arguments
            keyword arguments which the function :meth:`dolfin.plot` normally takes.
        """
        func = dl.Function(self.Vh)
        func.vector().zero()
        func.vector().set_local(values)
        p = dl.plot(func, **kwargs)
        self._label_coordinates(p)
        return p
    
    def _label_coordinates(self, p):
        pass


class FenicsContinuous1D(FenicsContinuous):

    def __init__(self,Vh):
        if Vh.mesh().geometry().dim()!=1:
            raise ValueError("'mesh' used in creating 'FenicsContinuous1D' object must be a one-dimensional mesh.")
        super().__init__(Vh)
    
    def _label_coordinates(self, p):
        plt.gca().set_xlabel('x')


class FenicsContinuous2D(FenicsContinuous):
    def __init__(self,Vh):
        if Vh.mesh().geometry().dim()!=2:
            raise ValueError("'mesh' used in creating 'FenicsContinuous2D' object must be a two-dimensional mesh.")
        super().__init__(Vh)
     
    def _label_coordinates(self, p):
        p.ax.set_xlabel('x')
        p.ax.set_ylabel('y')


class FenicsContinuous3D(FenicsContinuous):
    def __init__(self,Vh):
        raise NotImplementedError("'FenicsContinuous' object for 3D meshes is not implemented yet. 'mesh' needs to be a 1D or 2D mesh.")