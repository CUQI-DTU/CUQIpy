from abc import ABC, abstractmethod
import numpy as np
import numpy.matlib as matlib
import matplotlib.pyplot as plt
import math
from scipy.fftpack import dst, idst
import scipy.sparse as sparse

class Geometry(ABC):
    """A class that represents the geometry of the range, domain, observation, or other sets.
    """
    @property
    @abstractmethod
    def shape(self):
        pass

    @property
    def dim(self):
        if self.shape is None: return None
        return np.prod(self.shape)

    def plot(self, values, is_par=True, plot_par=False, **kwargs):
        """
        Plots a function over the set defined by the geometry object.
            
        Parameters
        ----------
        values : ndarray
            1D array that contains the values of the function degrees of freedom.

        is_par : Boolean, default True
            Flag to indicate whether the values are parameters or function values.
            True:  values are passed through the :meth:`par2fun` method.
            False: values are plotted directly.
        
        plot_par : Boolean, default False
            If true this method plots the parameters as a :class:`Discrete` geometry.
        """
        #Error check
        if plot_par and not is_par:
            raise Exception("Plot par is true, but is_par is false (parameters were not given)")

        if plot_par:
            geom = Discrete(self.dim) #dim is size of para (at the moment)
            return geom.plot(values,**kwargs)

        if is_par:
            values = self.par2fun(values)

        return self._plot(values,**kwargs)

    def plot_envelope(self, lo_values, hi_values, is_par=True, plot_par=False, **kwargs):
        """
        Plots an envelope from lower and upper bounds over the set defined by the geometry object.
            
        Parameters
        ----------
        lo_values : ndarray
            1D array that contains a lower bound of the function degrees of freedom.

        hi_values : ndarray
            1D array that contains an upper bound of the function degrees of freedom.

        is_par : Boolean, default True
            Flag to indicate whether the values are parameters or function values.
            True:  values are passed through the :meth:`par2fun` method.
            False: values are plotted directly.
        
        plot_par : Boolean, default False
            If true this method plots the parameters as a :class:`Discrete` geometry.
        """
        #Error check
        if plot_par and not is_par:
            raise Exception("Plot par is true, but is_par is false (parameters were not given)")
        
        if plot_par:
            geom = Discrete(self.dim) #dim is size of para (at the moment)
            return geom.plot_envelope(lo_values, hi_values,**kwargs)

        if is_par:
            lo_values = self.par2fun(lo_values)
            hi_values = self.par2fun(hi_values)

        return self._plot_envelope(lo_values,hi_values,**kwargs)

    def par2fun(self,par):
        """The parameter to function map used to map parameters to function values in e.g. plotting."""
        return par

    def fun2par(self,funvals):
        """The function to parameter map used to map function values back to parameters, if available."""
        raise NotImplementedError("fun2par not implemented. Must be implemented specifically for each geometry.")

    @abstractmethod
    def _plot(self):
        pass

    def _plot_envelope(self,*args,**kwargs):
        raise NotImplementedError("Plot envelope not implemented for {}. Use flag plot_par to plot envelope of parameters instead.".format(type(self)))
            
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

    def __eq__(self, obj):
        if not isinstance(obj, self.__class__): return False
        for key, value in vars(self).items():
            if not np.all(value ==vars(obj)[key]): return False 
        return True

class Continuous(Geometry, ABC):

    def __init__(self,grid=None, axis_labels=None, mapping=lambda x: x):
        self.axis_labels = axis_labels
        self.grid = grid
        self.mapping = mapping

    def _create_dimension(self, dim_grid):
        dim_grid_value_err_msg = "dim_grid should be int, tuple with one int element, list of numbers, 1D numpy.ndarray, or None"
        if dim_grid is None:
            return None

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

    def apply_map(self, p):
        return self.mapping( self.par2fun(p) )

    # plots the random field: "params", "KL", "mapped"
    def plot_mapped(self, p, is_par = False):
        plt.plot(self.grid, self.apply_map(p))
        
    @property
    def grid(self):
        return self._grid
    
    def fun2par(self,funvals):
        return funvals.ravel()

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

    def __init__(self,grid=None,axis_labels=['x'],**kwargs):
        super().__init__(grid, axis_labels,**kwargs)

    @property
    def shape(self):
        if self.grid is None: return None
        return self.grid.shape

    @Continuous.grid.setter
    def grid(self, value):
        self._grid = self._create_dimension(value)

    def _plot(self,values,*args,**kwargs):
        p = plt.plot(self.grid,values,*args,**kwargs)
        self._plot_config()
        return p

    def _plot_envelope(self, lo_values, up_values, **kwargs):
        default = {'color':'dodgerblue', 'alpha':0.25}
        for key in default:
            if (key not in kwargs.keys()):
                kwargs[key]  = default[key]
        return plt.fill_between(self.grid,up_values, lo_values, **kwargs)

    def _plot_config(self):
        if self.axis_labels is not None:
            plt.xlabel(self.axis_labels[0])


class Continuous2D(Continuous):

    def __init__(self,grid=None,axis_labels=['x','y']):
        super().__init__(grid, axis_labels)
            
    @property
    def shape (self):
        if self.grid is None: return None
        return (len(self.grid[0]), len(self.grid[1])) 

    @Continuous.grid.setter
    def grid(self, value):
        if value is None: self._grid = None
        else:
            if len(value)!=2:
                raise NotImplementedError("grid must be a 2D tuple of int values or arrays (list, tuple or numpy.ndarray) or combination of both")
            self._grid = (self._create_dimension(value[0]), self._create_dimension(value[1]))

    def _plot(self,values,plot_type='pcolor',**kwargs):
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

    def _plot(self,values, **kwargs):

        if ('linestyle' not in kwargs.keys()) and ('ls' not in kwargs.keys()):
            kwargs["linestyle"]  = ''
        
        if ('marker' not in kwargs.keys()):
            kwargs["marker"]  = 'o'

        self._plot_config() 
        return plt.plot(self._ids,values,**kwargs)

    def _plot_envelope(self, lo_values, up_values, **kwargs):
        self._plot_config()
        if 'fmt' in kwargs.keys():
            raise Exception("Argument 'fmt' cannot be passed by the user")

        default = {'color':'dodgerblue', 'fmt':'none' ,'capsize':3, 'capthick':1}
        for key in default:
            if (key not in kwargs.keys()):
                kwargs[key]  = default[key]

        #Convert to 1d numpy array to handle subtraction
        lo_values = np.array(lo_values).flatten()
        up_values = np.array(up_values).flatten()
        
        return plt.errorbar(self._ids, lo_values, 
                            yerr=np.vstack((np.zeros(len(lo_values)),up_values-lo_values)),
                            **kwargs)

    def _plot_config(self):
        plt.xticks(self._ids, self.variables)



class _DefaultGeometry(Continuous1D):
    def __init__(self,grid=None, axis_labels=['x']):
        super().__init__(grid, axis_labels)

    def __eq__(self, obj):
        if not isinstance(obj, (self.__class__,Continuous1D)): return False
        for key, value in vars(self).items():
            if not np.all(value == vars(obj)[key]): return False 
        return True

# class DiscreteField(Discrete):
#     def __init__(self, grid, cov_func, mean, std, trunc_term=100, axis_labels=['x']):
#         super().__init__(grid, axis_labels)

#         self.N = len(self.grid)
#         self.mean = mean
#         self.std = std
#         XX, YY = np.meshgrid(self.grid, self.grid, indexing='ij')
#         self.Sigma = cov_func(XX, YY)
#         self.L = np.linalg.chol(self.Sigma)

#     def par2fun(self, p):
#         return self.mean + self.L.T@p
    
class KLExpansion(Continuous1D):
    '''
    class representation of the random field in  the sine basis
    alpha = sum_i p * (1/i)^decay * sin(i*L*x/pi)
    L is the length of the interval
    '''
    
    # init function defining paramters for the KL expansion
    def __init__(self, grid, axis_labels=['x'],**kwargs):
        
        super().__init__(grid, axis_labels,**kwargs)

        self.decay_rate = 2.5 # decay rate of KL
        self.normalizer = 12. # normalizer factor
        eigvals = np.array( range(1,self.N+1) ) # KL eigvals
        self.coefs = 1/np.float_power( eigvals,self.decay_rate )


    # computes the real function out of expansion coefs
    def par2fun(self,p):
        modes = p*self.coefs/self.normalizer
        real = idst(modes)/2
        return real
    
<<<<<<< HEAD
    def fun2par(self,funvals):
        """The function to parameter map used to map function values back to parameters, if available."""
        raise NotImplementedError("fun2par not implemented. ")
=======
>>>>>>> fc1f82b (Remove unused stuff in KLExpansion)

class CustomKL(Continuous1D):
    def __init__(self, grid, cov_func, mean, std, trunc_term=100, axis_labels=['x'],**kwargs):
        super().__init__(grid, axis_labels,**kwargs)

        self.N = len(self.grid)
        self.mean = mean
        self.std = std
        self.Nystrom( grid, cov_func, std, trunc_term, int(2*self.N) )

    def par2fun(self, p):
        return self.mean + ((self.eigvec@np.diag(np.sqrt(self.eigval))) @ p)
    
    def fun2par(self,funvals):
        """The function to parameter map used to map function values back to parameters, if available."""
        raise NotImplementedError("fun2par not implemented. ")
    

    def Nystrom(self, xnod, C_nu, sigma, M, N_GL):
        # xnod: points at which the field is realized (from geometry PDE)
        # C_nu: lambda function with the covariance kernel (no variance multiplied)
        # sigma: standard deviation of the field
        # M: dimension of the field (KL truncation)
        # N_GL: gauss-legendre points for the integration
        ### return eigenpairs of an arbitrary 1D correlation kernel

        # domain data    
        n = xnod.size
        a = (xnod[-1] - xnod[0])/2     # scale
        
        # compute the Gauss-Legendre abscissas and weights
        #xi, w   = quad.Gauss_Legendre(N_GL) 
        xi, w = np.polynomial.legendre.leggauss(N_GL)

        # transform nodes and weights to [0, L]
        xi_s = a*xi + a
        w_s = a*w
        
        # compute diagonal matrix 
        D = sparse.spdiags(np.sqrt(w_s), 0, N_GL, N_GL).toarray()
        S1 = matlib.repmat(np.sqrt(w_s).reshape(1, N_GL), N_GL, 1)
        S2 = matlib.repmat(np.sqrt(w_s).reshape(N_GL, 1), 1, N_GL)
        S = S1 * S2
        
        # compute covariance matrix 
        Sigma_nu = np.zeros((N_GL, N_GL))
        for i in range(N_GL):
            for j in range(N_GL):
                if i != j:
                    Sigma_nu[i,j] = C_nu(xi_s[i], xi_s[j])
                else:
                    Sigma_nu[i,j] = sigma**2   # correct the diagonal term
                      
        # solve the eigenvalue problem
        A = Sigma_nu * S                             # D_sqrt*Sigma_nu*D_sqrt
        L, h = np.linalg.eig(A)                         # solve eigenvalue problem
        # L, h = sp.sparse.linalg.eigsh(A, M, which='LM')   # np.linalg.eig(A)         
        idx = np.argsort(-np.real(L))                  # index sorting descending
        
        # order the results
        eigval = np.real(L[idx])
        h = h[:,idx]
        
        # take the M values
        eigval = eigval[:M]
        h = np.real(h[:,:M])
        
        # replace for the actual eigenvectors
        phi = np.linalg.solve(D, h)
        
        # Nystrom's interpolation formula
        # recompute covariance matrix on partition nodes and quadrature nodes
        Sigma_nu = np.zeros((n, N_GL))
        for i in range(n):
            for j in range(N_GL):
                Sigma_nu[i,j] = C_nu(xnod[i], xi_s[j])
                          
        M1 = Sigma_nu * np.matlib.repmat(w_s.reshape(N_GL,1), 1, n).T
        M2 = phi @ np.diag(1/eigval)
        eigvec = M1 @ M2

        # normalize eigenvectors (not necessary) integrate to 1
        #norm_fact = np.zeros((M,1))
        #for i in range(M):
        #    norm_fact[i] = np.sqrt(np.trapz(eigvec[:,i]**2, xnod))
        #eigvec = eigvec/np.matlib.repmat(norm_fact, 1, n)             
        self.eigval = eigval
        self.eigvec = eigvec


class StepExpansion(Continuous1D):
    '''
    class representation of the step random field with 3 intermidiate steps
    '''
    def __init__(self, grid, axis_labels=['x'], **kwargs):

        super().__init__(grid, axis_labels,**kwargs)

        self.N = len(self.grid) # number of modes
        self.p = np.zeros(4)
        self.L = grid[-1]
        #self.dx = np.pi/(self.N+1)
        #self.x = np.linspace(self.dx,np.pi,N,endpoint=False)
        self.axis_labels = axis_labels

    def par2fun(self, p):
        self.real = np.zeros_like(self.grid)        
        idx = np.where( (self.grid>0.2*self.L)&(self.grid<=0.4*self.L) )
        self.real[idx[0]] = p[0]
        idx = np.where( (self.grid>0.4*self.L)&(self.grid<=0.6*self.L) )
        self.real[idx[0]] = p[1]
        idx = np.where( (self.grid>0.6*self.L)&(self.grid<=0.8*self.L) )
        self.real[idx[0]] = p[2]
        return self.real
    
    def fun2par(self,funvals):
        """The function to parameter map used to map function values back to parameters, if available."""
        raise NotImplementedError("fun2par not implemented. ")
    
    @property
    def shape(self):
        return 3
    
    '''
    def plot(self,values,*args,**kwargs):
        p = plt.plot(values,*args,**kwargs)
        self._plot_config()
        return p

    def _plot_config(self):
        if self.axis_labels is not None:
            plt.xlabel(self.axis_labels[0])
            '''