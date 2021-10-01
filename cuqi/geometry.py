from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.fftpack import dst, idst

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
        dim_grid_value_err_msg = "dim_grid should be int, tuple with one int element, list of numbers, or 1D numpy.ndarray"
        if isinstance(dim_grid,tuple) and len(dim_grid)==1:
            dim_grid = dim_grid[0]

        if isinstance(dim_grid,(int,np.integer)):
            dim_grid = np.arange(dim_grid)
        elif isinstance(dim_grid,(list,np.ndarray)):
            dim_grid = np.array(dim_grid)
            if len(dim_grid.shape)!=1:
                raise ValueError(dim_grid_value_err_msg)
        else:
            raise ValueError(dim_grid_value_err_msg)
        return dim_grid

    @property
    def grid(self):
        return self._grid

class Continuous1D(Continuous):
    """A class that represents a continuous 1D geometry.

    Parameters
    -----------
    grid : int, tuple, list or numpy.ndarray
        1D array of node coordinates in a 1D grid (list or numpy.ndarray), or number of nodes (int or tuple with one int element) in the grid. In the latter case, a default grid with unit spacing and coordinates 0,1,2,... will be created.

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

    def plot_envelope(self, lo_values, up_values, **kwargs):
        default = {'color':'dodgerblue', 'alpha':0.25}
        for key in default:
            if (key not in kwargs.keys()):
                kwargs[key]  = default[key]
        return plt.fill_between(self.grid,up_values, lo_values, **kwargs)

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


class Discrete(Geometry):

    def __init__(self,variables):       
        self.variables = variables

    @property
    def shape(self):
        return (len(self.variables),)

    @property
    def variables(self):
        return self._variables

    @variables.setter
    def variables(self, value):
        variables_value_err_msg = "variables should be int, or list of strings"
        if isinstance(value,(int,np.integer)):
            value = ['v'+str(var) for var in range(value)]
        elif isinstance(value,list): 
            for var in value: 
                if not isinstance(var,str):
                    raise ValueError(variables_value_err_msg)
        else:
            raise ValueError(variables_value_err_msg) 
        self._variables = value
        self._ids = range(self.dim)

    def plot(self,values, **kwargs):

        if ('linestyle' not in kwargs.keys()) and ('ls' not in kwargs.keys()):
            kwargs["linestyle"]  = ''
        
        if ('marker' not in kwargs.keys()):
            kwargs["marker"]  = 'o'

        self._plot_config() 
        return plt.plot(self._ids,values,**kwargs)

    def plot_envelope(self, lo_values, up_values, **kwargs):
        self._plot_config()
        if 'fmt' in kwargs.keys():
            raise Exception("Argument 'fmt' cannot be passed by the user")

        default = {'color':'dodgerblue', 'fmt':'none' ,'capsize':3, 'capthick':1}
        for key in default:
            if (key not in kwargs.keys()):
                kwargs[key]  = default[key]
        
        return plt.errorbar(self._ids, lo_values, 
                            yerr=np.vstack((np.zeros(len(lo_values)),up_values-lo_values)),
                            **kwargs)

    def _plot_config(self):
        plt.xticks(self._ids, self.variables)


class KLField(Geometry):
    '''
    class representation of the random field in  the sine basis
    alpha = sum_i p * (1/i)^decay * sin(ix)
    '''
    
    # init function defining paramters for the KL expansion
    def __init__(self, N, axis_labels=['x']):
        self.N = N # number of modes
        self.modes = np.zeros(N) # vector of expansion coefs
        self.real = np.zeros(N) # vector of real values
        self.decay_rate = 2.5 # decay rate of KL
        self.c = 12. # normalizer factor
        self.coefs = np.array( range(1,self.N+1) ) # KL eigvals
        self.coefs = 1/np.float_power( self.coefs,self.decay_rate )

        self.p = np.zeros(self.N) # random variables in KL

        self.axis_labels = axis_labels

    # computes the real function out of expansion coefs
    def to_function(self,p):
        self.modes = p*self.coefs/self.c
        self.real = idst(self.modes)/2
        return self.real
    
    @property
    def shape(self):
        return self.N
    
    def plot(self,values,*args,**kwargs):
        p = plt.plot(values,*args,**kwargs)
        self._plot_config()
        return p

    def _plot_config(self):
        if self.axis_labels is not None:
            plt.xlabel(self.axis_labels[0])

class StepField(Geometry):
    '''
    class representation of the step random field with 3 intermidiate steps
    '''
    def __init__(self,N, axis_labels=['x']):
        self.N = N
        self.p = np.zeros(4)
        self.dx = np.pi/(self.N+1)
        self.x = np.linspace(self.dx,np.pi,N,endpoint=False)

        self.axis_labels = axis_labels

    def to_function(self, p):
        self.real = np.zeros_like(self.x)
        
        idx = np.where( (self.x>0.2*np.pi)&(self.x<=0.4*np.pi) )
        self.real[idx[0]] = p[0]
        idx = np.where( (self.x>0.4*np.pi)&(self.x<=0.6*np.pi) )
        self.real[idx[0]] = p[1]
        idx = np.where( (self.x>0.6*np.pi)&(self.x<=0.8*np.pi) )
        self.real[idx[0]] = p[2]
        return self.real
    
    @property
    def shape(self):
        return 3
    
    def plot(self,values,*args,**kwargs):
        p = plt.plot(values,*args,**kwargs)
        self._plot_config()
        return p

    def _plot_config(self):
        if self.axis_labels is not None:
            plt.xlabel(self.axis_labels[0])


        