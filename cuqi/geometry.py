from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import math

class Geometry(ABC):
    """A class that represents the geometry of the range, domain, observation, or other sets.
    """

    @abstractmethod
    def plot(self,values):
        """
        Plots a function over the set defined by the geometry object.
            
        Parameters
        ----------
        values : ndarray
            1D array that contains the values of the function degrees of freedom.
        """
        pass

    @abstractmethod
    def _pre_plot(self,values):
        """
        A method that implements any default configuration or set up that needs to occur before ploting. This method is to be called inside any 'plot_' method.
        """
        pass

class Continuous1D(Geometry):

    def __init__(self,dim,grid=None,labels=['x']):
        self.labels = labels
        if len(dim)==1:
            if grid is None:
                self.grid = np.arange(dim[0])
            else:
                self.grid = grid
        else:
            raise NotImplementedError("Cannot init 1D geometry with spatial dimension > 1")

    def plot(self,values,*args,**kwargs):
        values = self._pre_plot(values)
        return plt.plot(self.grid,values,*args,**kwargs)

    def _pre_plot(self, values):
        if self.labels is not None:
            plt.xlabel(self.labels[0])
        return values


class Continuous2D(Geometry):

    def __init__(self,dim,grid=None,labels=['x','y']):
        self.labels = labels
        if len(dim)!=2:
            raise NotImplemented("Cannot init 2D geometry with spatial dimension != 2")

        if grid is None:
            self.grid = np.meshgrid( np.arange(dim[0]), np.arange(dim[1]) )
        else:
            self.grid = grid

    def plot(self,values,**kwargs):
        """
        Overrides :meth:`cuqi.geometry.Geometry.plot`. See :meth:`cuqi.geometry.Geometry.plot` for description  and definition of the parameter `values`.
        
        Parameters
        -----------
        kwargs : keyword arguments
            keyword arguments which the function :meth:`matplotlib.pyplot.pcolor` normally takes.
        """
        values = self._pre_plot(values)
        ims = []
        for i, axis in enumerate(plt.gcf().axes):
            ims.append(axis.pcolor(self.grid[0],self.grid[1],values[...,i].reshape(self.grid[0].shape),
                          **kwargs))
        return ims
    
    def plot_contour(self,values,**kwargs):
        values = self._pre_plot(values)
        ims = []
        for i, axis in enumerate(plt.gcf().axes):
            ims.append(axis.contour(self.grid[0],self.grid[1],values[...,i].reshape(self.grid[0].shape),
                          **kwargs))
        return ims

    def plot_contourf(self,values,**kwargs):
        values = self._pre_plot(values)
        ims = []
        for i, axis in enumerate(plt.gcf().axes):
            ims.append(axis.contourf(self.grid[0],self.grid[1],values[...,i].reshape(self.grid[0].shape),
                          **kwargs))
        return ims
    
    def _pre_plot(self,values):
        if len(values.shape) == 3 or\
             (len(values.shape) == 2 and values.shape[0]== len(self.grid[0].flat)): #TODO: change len(self.domain_geometry.grid) to self.domain_geometry.ndofs 
            self._create_subplots(values.shape[-1])
        else:
            self._create_subplots(1)
            values = values[..., np.newaxis]

        for i, axis in enumerate(plt.gcf().axes):
            if self.labels is not None:
                axis.set_xlabel(self.labels[0])
                axis.set_ylabel(self.labels[1])
            axis.set_aspect('equal')
      
        return values

    def _create_subplots(self,N):
        Nx = math.ceil(np.sqrt(N))
        Ny = Nx
        fig = plt.gcf()
        fig.set_size_inches(fig.bbox_inches.corners()[3][0]*Nx, fig.bbox_inches.corners()[3][1]*Ny)
        for i in range(N):
            subplot_id = Ny*100+ Nx*10 + (i+1)
            fig.add_subplot(subplot_id)