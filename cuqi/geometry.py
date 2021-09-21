from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import math

class Geometry(ABC):
    """A class that represents the geometry of the range, domain, observation, or other sets.
    """
    @property
    @abstractmethod
    def shape(self):
        pass

    @property
    def dim(self):
        return np.prod(self.shape)

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

    def _plot_config(self,values):
        """
        A method that implements any default configuration for the plots. This method is to be called inside any 'plot_' method.
        """
        pass

    def _create_subplot_list(self,values,subplots=True):
        Ns = values.shape[-1]
        Nx = math.ceil(np.sqrt(Ns))
        Ny = Nx
        subplot_ids = []
        fig = plt.gcf()
        if subplots: fig.set_size_inches(fig.bbox_inches.corners()[3][0]*Nx, fig.bbox_inches.corners()[3][1]*Ny)

        for i in range(Ny):
            for j in range(Nx):
                subplot_id = i*Nx+j+1
                if subplot_id > Ns: 
                    continue 
                subplot_ids.append((Ny,Nx,subplot_id))
        return subplot_ids

class Continuous(Geometry, ABC):

    def __init__(self,grid,axis_labels):
        self.axis_labels = axis_labels
        self.grid = grid

    def _create_dimension(self, dim_grid):
        if isinstance(dim_grid,(int,np.integer)):
            dim_grid = np.arange(dim_grid)
        elif hasattr(dim_grid,'__len__'):
            dim_grid = np.array(dim_grid)
            if len(dim_grid.shape)!=1:
                raise ValueError("dim_grid must be a 1D row array")
        else:
            raise ValueError("dim_grid should be int, list, numpy.ndarray or tuple")
        return dim_grid

    @property
    def grid(self):
        return self._grid

class Continuous1D(Continuous):
    """A class that represents a continuous 1D geometry.

    Parameters
    -----------
    grid : int, list, tuple or numpy.ndarray
        1D array of node coordinates in a 1D grid (list, tuple or numpy.ndarray) or number
        of nodes (int) in the grid. If grid is of type int, a default grid
        with unit spacing and coordinates 0,1,2,...(grid-1) will be created.

    Attributes
    -----------
    grid : numpy.ndarray
        1D array of node coordinates in a 1D grid
    """

    def __init__(self,grid,axis_labels=['x']):
        super().__init__(grid, axis_labels)

    @property
    def shape(self):
        return self.grid.shape

    @Continuous.grid.setter
    def grid(self, value):
        self._grid = self._create_dimension(value)

    def plot(self,values,*args,**kwargs):
        p = plt.plot(self.grid,values,*args,**kwargs)
        self._plot_config()
        return p

    def _plot_config(self):
        if self.axis_labels is not None:
            plt.xlabel(self.axis_labels[0])


class Continuous2D(Continuous):

    def __init__(self,grid,axis_labels=['x','y']):
        super().__init__(grid, axis_labels)
            
    @property
    def shape (self):
        return (len(self.grid[0]), len(self.grid[1])) 

    @Continuous.grid.setter
    def grid(self, value):
        if len(value)!=2:
            raise NotImplementedError("grid must be a 2D tuple of int values or arrays (list, tuple or numpy.ndarray) or combination of both")
        self._grid = (self._create_dimension(value[0]), self._create_dimension(value[1]))

    def plot(self,values,plot_type='pcolor',**kwargs):
        """
        Overrides :meth:`cuqi.geometry.Geometry.plot`. See :meth:`cuqi.geometry.Geometry.plot` for description  and definition of the parameter `values`.
        
        Parameters
        -----------
        plot_type : str
            type of the plot. If plot_type = 'pcolor', :meth:`matplotlib.pyplot.pcolor` is called, if plot_type = 'contour', :meth:`matplotlib.pyplot.contour` is called, and if `plot_type` = 'contourf', :meth:`matplotlib.pyplot.contourf` is called, 

        kwargs : keyword arguments
            keyword arguments which the methods :meth:`matplotlib.pyplot.pcolor`, :meth:`matplotlib.pyplot.contour`, or :meth:`matplotlib.pyplot.contourf`  normally take, depending on the value of the parameter `plot_type`.
        """
        if plot_type == 'pcolor': 
            plot_method = plt.pcolor
        elif plot_type == 'contour':
            plot_method = plt.contour
        elif plot_type == 'contourf':
            plot_method = plt.contourf
        else:
            raise ValueError(f"unknown value: {plot_type} of the parameter 'plot_type'")
        
        values = self._process_values(values)
        subplot_ids = self._create_subplot_list(values)
        ims = []
        for rows,cols,subplot_id in subplot_ids:
            plt.subplot(rows,cols,subplot_id); 
            ims.append(plot_method(self.grid[0],self.grid[1],values[...,subplot_id-1].reshape(self.shape[::-1]),
                          **kwargs))
        self._plot_config()
        return ims

    def plot_pcolor(self,values,**kwargs):
        return self.plot(values,plot_type='pcolor',**kwargs)

    def plot_contour(self,values,**kwargs):
        return self.plot(values,plot_type='contour',**kwargs)

    def plot_contourf(self,values,**kwargs):
       return self.plot(values,plot_type='contourf',**kwargs)
    
    def _process_values(self,values):
        if len(values.shape) == 3 or\
             (len(values.shape) == 2 and values.shape[0]== self.dim):  
            pass
        else:
            values = values[..., np.newaxis]
        return values

    def _plot_config(self):
        for i, axis in enumerate(plt.gcf().axes):
            if self.axis_labels is not None:
                axis.set_xlabel(self.axis_labels[0])
                axis.set_ylabel(self.axis_labels[1])
            axis.set_aspect('equal')