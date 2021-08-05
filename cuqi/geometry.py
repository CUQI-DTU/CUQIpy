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

    def _create_subplots(self,N):
        Nx = math.ceil(np.sqrt(N))
        Ny = Nx
        fig = plt.gcf()
        fig.set_size_inches(fig.bbox_inches.corners()[3][0]*Nx, fig.bbox_inches.corners()[3][1]*Ny)
        for i in range(N):
            fig.add_subplot(Ny,Nx,i+1)

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
        return plt.plot(self.grid,values,*args,**kwargs)

class Continuous2D(Geometry):

    def __init__(self,dim,grid=None):
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
        self._label_coordinates()
        return plt.pcolor(self.grid[0],self.grid[1],values.reshape(self.grid[0].shape),
                          **kwargs)
    
    def plot_contour(self,values,**kwargs):
        self._label_coordinates()
        return plt.contour(self.grid[0],self.grid[1],values.reshape(self.grid[0].shape),
                           **kwargs)

    def plot_contourf(self,values,**kwargs):
        self._label_coordinates()
        return plt.contourf(self.grid[0],self.grid[1],values.reshape(self.grid[0].shape),
                            **kwargs)
    
    def _label_coordinates(self):
        plt.xlabel('x')
        plt.ylabel('y')
