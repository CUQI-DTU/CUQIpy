import numpy as np
import matplotlib.pyplot as plt
from cuqi.diagnostics import Geweke
from cuqi.geometry import _DefaultGeometry, Continuous2D, Image2D
from copy import copy
import arviz # Plotting tool

class CUQIarray(np.ndarray):
    """
    A class to represent data arrays, subclassed from numpy array, along with geometry and plotting

    Parameters
    ----------
    input_array : ndarray
        A numpy array holding the parameter or function values. 
    
    is_par : bool, default True
        Boolean flag whether input_array is to be interpreted as parameter (True) or function values (False).

    geometry : cuqi.geometry.Geometry, default None
        Contains the geometry related of the data

    Attributes
    ----------
    funvals : CUQIarray
        Returns itself as function values.

    parameters : CUQIarray
        Returns itself as parameters.

    Methods
    ----------
    :meth:`plot`: Plots the data as function or parameters.
    """

    def __repr__(self) -> str: 
        return "CUQIarray: NumPy array wrapped with geometry.\n" + \
               "---------------------------------------------\n\n" + \
            "Geometry:\n {}\n\n".format(self.geometry) + \
            "Parameters:\n {}\n\n".format(self.is_par) + \
            "Array:\n" + \
            super().__repr__()

    def __new__(cls, input_array, is_par=True, geometry=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.is_par = is_par
        obj.geometry = geometry
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.is_par = getattr(obj, 'is_par', True)
        self.geometry = getattr(obj, 'geometry', None)

    @property
    def funvals(self):
        if self.is_par is True:
            vals = self.geometry.par2fun(self)
        else:
            vals = self

        if isinstance(vals, np.ndarray):
            return type(self)(vals,is_par=False,geometry=self.geometry) #vals.view(np.ndarray)
        else: 
            return vals  

    @property
    def parameters(self):
        if self.is_par is False:
            vals = self.geometry.fun2par(self)
        else:
            vals = self
        return type(self)(vals,is_par=True,geometry=self.geometry)
    
    def plot(self, plot_par=False, **kwargs):
        if plot_par:
            return self.geometry.plot(self.parameters, plot_par=plot_par, is_par=True, **kwargs)
        else:
            return self.geometry.plot(self.funvals, is_par=False, **kwargs)


class Data(object):
    """
    An container type object to represent data objects equipped with geometry.
    """

    def __init__(self, parameters=None, geometry=None, funvals=None):
        
        # Allow setting either parameter or function values, but not both.
        # If both provided, function values take precedence (to be confirmed).
        if parameters is not None and funvals is not None:
            parameters = None
        
        if parameters is not None:
            self.parameters = parameters
        
        if funvals is not None:
            self.funvals = funvals

        self.geometry = geometry
        
    def plot(self, plot_par=False, **kwargs):
        if plot_par:
            self.geometry.plot(self.parameters, plot_par=plot_par, is_par=True, **kwargs)
        else:
            self.geometry.plot(self.funvals, is_par=False, **kwargs)
    
    @property
    def parameters(self):
        return self._parameters
    
    @parameters.setter
    def parameters(self, value):
        self._parameters = value
        self.has_parameters = True
        self.has_funvals = False
    
    @property
    def funvals(self):
        if self.has_funvals:
            return self._funvals
        else:
            return self.geometry.par2fun(self.parameters)
    
    @funvals.setter
    def funvals(self, value):
        self.has_funvals = True
        self.has_parameters = False
        self._funvals = value


class Samples(object):
    """
    An object used to store samples from distributions. 

    Parameters
    ----------
    samples : ndarray
        Contains the raw samples as a numpy array indexed by the last axis of the array.

    geometry : cuqi.geometry.Geometry, default None
        Contains the geometry related of the samples

    Attributes
    ----------
    shape : tuple
        Returns the shape of samples.

    Ns : int
        Returns the number of samples

    Methods
    ----------
    :meth:`plot`: Plots one or more samples.
    :meth:`plot_ci`: Plots a credibility interval for the samples.
    :meth:`plot_mean`: Plots the mean of the samples.
    :meth:`plot_std`: Plots the std of the samples.
    :meth:`plot_chain`: Plots all samples of one or more variables (MCMC chain).
    :meth:`hist_chain`: Plots histogram of all samples of a single variable (MCMC chain).
    :meth:`burnthin`: Removes burn-in and thins samples.
    :meth:`diagnostics`: Conducts diagnostics on the chain.
    """
    def __init__(self, samples, geometry=None):
        self.samples = samples
        self.geometry = geometry

    @property
    def shape(self):
        return self.samples.shape

    @property
    def Ns(self):
        """Return number of samples"""
        return self.samples.shape[-1]

    @property
    def geometry(self):
        if self._geometry is None:
            self._geometry = _DefaultGeometry(grid=np.prod(self.samples.shape[:-1]))
        return self._geometry

    @geometry.setter
    def geometry(self,inGeometry):
        self._geometry = inGeometry

    def burnthin(self, Nb, Nt=1):
        """
        Remove burn-in and thin samples. 
        The burnthinned samples are returned as a new Samples object.
        
        Parameters
        ----------
        Nb : int
            Number of samples to remove as burn-in from the start of the chain.
        
        Nt : int
            Thin samples by selecting every Nt sample in the chain (after burn-in)

        Example
        ----------
        # Remove 100 samples burn in and select every 2nd sample after burn-in
        # Store as new samples object
        S_burnthin = S.burnthin(100,2) 

        # Same thing as above, but replace existing samples object
        # (the burn-in and thinned samples are lost)
        S = S.burnthin(100,2) 
        """
        if Nb>=self.Ns:
            raise ValueError(f"Number of burn-in {Nb} is greater than or equal number of samples {self.Ns}")
        new_samples = copy(self)
        new_samples.samples = self.samples[...,Nb::Nt]
        return new_samples

    def plot_mean(self,*args,**kwargs):
        """Plot pointwise mean of the samples

        Positional and keyword arguments are passed to the underlying `self.geometry.plot` method.
        See documentation of `self.geometry` for options.
        """
        mean = self.mean()

        # Plot mean according to geometry
        ax =  self.geometry.plot(mean, *args, **kwargs)
        plt.title('Sample mean')
        return ax

    def plot_variance(self,*args,**kwargs):
        """Plot pointwise variance of the samples

        Positional and keyword arguments are passed to the underlying `self.geometry.plot` method.
        See documentation of `self.geometry` for options.
        """
        variance = self.variance()

        # Plot variance according to geometry
        ax = self.geometry.plot(variance, *args, **kwargs)
        plt.title('Sample variance')
        return ax

    def mean(self):
        """Compute mean of the samples"""
        return np.mean(self.samples, axis=-1)

    def variance(self):
        """Compute pointwise variance of the samples"""
        return np.var(self.samples, axis=-1)

    def plot_std(self,*args,**kwargs):
        """Plot pointwise standard deviation of the samples

        Positional and keyword arguments are passed to the underlying `self.geometry.plot` method.
        See documentation of `self.geometry` for options.
        """

        # Compute std assuming samples are index in last dimension of nparray
        std = np.std(self.samples,axis=-1)

        # Plot mean according to geometry
        ax = self.geometry.plot(std, *args, **kwargs)
        plt.title('Sample standard deviation')
        return ax

    def plot(self,sample_indices=None,*args,**kwargs):
        Ns = self.Ns
        Np = 5 # Number of samples to plot if Ns > 5
        
        if sample_indices is None:
            if Ns>Np: print("Plotting {} randomly selected samples".format(Np))
            sample_indices = self._select_random_indices(Np, Ns)
        
        return self.geometry.plot(self.samples[:,sample_indices],*args,**kwargs)


    def plot_chain(self, variable_indices=None, *args, **kwargs):
        dim = self.geometry.dim
        Nv = 5 # Max number of variables to plot if none are chosen
        # If no variables are given we randomly select some at random
        if variable_indices is None:
            if Nv<dim: print(f"Selecting {Nv} randomly chosen variables")
            variable_indices = self._select_random_indices(Nv, dim)
        if 'label' in kwargs.keys():
            raise Exception("Argument 'label' cannot be passed by the user")
        variables = np.array(self.geometry.variables) #Convert to np array for better slicing
        variables = variables[variable_indices].flatten()
        lines = plt.plot(self.samples[variable_indices,:].T,*args,**kwargs)
        plt.legend(variables)
        return lines
    
    def hist_chain(self,variable_indices,*args,**kwargs):
        if 'label' in kwargs.keys():
            raise Exception("Argument 'label' cannot be passed by the user")
        variables = np.array(self.geometry.variables) #Convert to np array for better slicing
        variables = variables[variable_indices].flatten()
        n, bins, patches = plt.hist(self.samples[variable_indices,:].T,*args,**kwargs)
        plt.legend(variables)
        return patches

    def plot_ci(self,percent=95,exact=None,*args,plot_envelope_kwargs={},**kwargs):
        """
        Plots the credibility interval for the samples according to the geometry.

        Parameters
        ---------
        percent : int
            The percent credibility to plot (i.e. 95, 99 etc.)
        
        exact : ndarray, default None
            The exact value (for comparison)

        plot_envelope_kwargs : dict, default {}
            Keyword arguments for the plot_envelope method
        
        """
        
        # Compute statistics
        mean = np.mean(self.samples,axis=-1)
        lb = (100-percent)/2
        up = 100-lb
        lo_conf, up_conf = np.percentile(self.samples, [lb, up], axis=-1)

        #Extract plotting keywords and put into plot_envelope
        if len(plot_envelope_kwargs)==0:
            pe_kwargs={}
        else:
            pe_kwargs = plot_envelope_kwargs
        if "is_par"   in kwargs.keys(): pe_kwargs["is_par"]  =kwargs.get("is_par")
        if "plot_par" in kwargs.keys(): pe_kwargs["plot_par"]=kwargs.get("plot_par")   

        if type(self.geometry) is Continuous2D or type(self.geometry) is Image2D:
            plt.figure()
            #fig.add_subplot(2,2,1)
            self.geometry.plot(mean, *args, **kwargs)
            plt.title("Sample mean")
            if exact is not None:
                #fig.add_subplot(2,2,3)
                plt.figure()
                self.geometry.plot(exact, *args, **kwargs)
                plt.title("Exact")
            #fig.add_subplot(2,2,2)
            plt.figure()
            self.geometry.plot(up_conf-lo_conf)
            plt.title("Credibility interval (Upper minus lower)")
            plt.figure()
            self.geometry.plot(up_conf)
            plt.title("Upper credibility interval limit")
            #fig.add_subplot(2,2,4)
            plt.figure()
            self.geometry.plot(lo_conf)
            plt.title("Lower credibility interval limit")
        else:
            lci = self.geometry.plot_envelope(lo_conf, up_conf,color='dodgerblue',**pe_kwargs)
            
            lmn = self.geometry.plot(mean,*args,**kwargs)
            if exact is not None: #TODO: Allow exact to be defined in different space than mean?
                if isinstance(exact, CUQIarray):
                    lex = exact.plot(*args,**kwargs)
                else:
                    lex = self.geometry.plot(exact,*args,**kwargs)
                plt.legend([lmn[0], lex[0], lci],["Mean","Exact","Credibility Interval"])
            else:
                plt.legend([lmn[0], lci],["Mean","Credibility Interval"])

    def diagnostics(self):
        # Geweke test
        Geweke(self.samples.T)

    def plot_autocorrelation(self, variable_indices=None, max_lag=None, combined=True, **kwargs):
        """Plot the autocorrelation function of one or more variables in a single chain.

        Parameters
        ----------
        variable_indices : list, optional
            List of variable indices to plot the autocorrelation for. If no input is given and less than 5 variables exist all are plotted and with more 5 are randomly chosen.

        max_lag : int, optional
            Maximum lag to calculate autocorrelation. Defaults to 100 or number of samples,
            whichever is smaller.

        combined: bool, default=True
            Flag for combining multiple chains into a single chain. If False, chains will be
            plotted separately. Note multiple chains are not fully supported yet.

        Any remaining keyword arguments will be passed to the arviz plotting tool.
        See https://arviz-devs.github.io/arviz/api/generated/arviz.plot_autocorr.html.

        Returns
        -------
        axes: matplotlib axes or bokeh figures
        """
        dim = self.geometry.dim
        Nv = 5 # Max number of variables to plot if none are chosen

        # If no variables are given we randomly select some at random
        if variable_indices is None:
            if Nv<dim: print("Selecting 5 randomly chosen variables")
            variable_indices = self._select_random_indices(Nv, dim)

        # Convert to arviz InferenceData object
        datadict = self.to_arviz_inferencedata(variable_indices)
        
        # Plot autocorrelation using arviz
        axis = arviz.plot_autocorr(datadict, max_lag=max_lag, combined=combined, **kwargs)

        return axis

    def plot_trace(self, variable_indices=None, combined=True, tight_layout=True, **kwargs):
        """Creates a traceplot of the samples consisting of 1) a histogram/density plot of the samples and 2) an MCMC chain plot.
        
        Parameters
        ----------
        variable_indices : list, optional
            List of variable indices to plot the autocorrelation for. If no input is given and less than 5 variables exist all are plotted and with more 5 are randomly chosen.

        combined : bool, default=True
            Flag for combining multiple chains into a single chain. If False, chains will be
            plotted separately. Note multiple chains are not fully supported yet.        

        tight_layout: bool, default=True
            Improves the layout of the traceplot for multiple variables by calling `plt.tight_layout()`.
            Set to False if this causes issues.

        Any remaining keyword arguments will be passed to the arviz plotting tool.
        See https://arviz-devs.github.io/arviz/api/generated/arviz.plot_trace.html.

        Returns
        -------
        axes: matplotlib axes or bokeh figures  

        """
        dim = self.geometry.dim
        Nv = 5 # Max number of variables to plot if none are chosen

        # If no variables are given we randomly select some at random
        if variable_indices is None:
            if Nv<dim: print("Selecting 5 randomly chosen variables")
            variable_indices = self._select_random_indices(Nv, dim)

        # Convert to arviz InferenceData object
        datadict = self.to_arviz_inferencedata(variable_indices)

        # Plot using arviz
        ax =  arviz.plot_trace(datadict, combined=combined, **kwargs)

        # Improves subplot spacing
        if tight_layout: plt.tight_layout() 

        return ax

    def plot_pair(self, variable_indices=None, kind="scatter", marginals=False, **kwargs):
        """Plot marginals using a scatter, kde and/or hexbin matrix.
        
        Parameters
        ----------
        variable_indices : list, optional
            List of variable indices to plot the autocorrelation for. If no input is given and less than 5 variables exist all are plotted and with more 5 are randomly chosen.

        kind : str or List[str], default="scatter"
            Type of plot to display (scatter, kde and/or hexbin)

        marginals: bool, optional, default=False
            If True pairplot will include marginal distributions for every variable

        Any remaining keyword arguments will be passed to the arviz plotting tool.
        See https://arviz-devs.github.io/arviz/api/generated/arviz.plot_pair.html.

        Returns
        -------
        axes: matplotlib axes or bokeh figures  
        
        """
        dim = self.geometry.dim
        Nv = 5 # Max number of variables to plot if none are chosen

        # If no variables are given we randomly select some at random
        if variable_indices is None:
            if Nv<dim: print("Selecting 5 randomly chosen variables")
            variable_indices = self._select_random_indices(Nv, dim)

        # Convert to arviz InferenceData object
        datadict = self.to_arviz_inferencedata(variable_indices)

        ax =  arviz.plot_pair(datadict, kind=kind, marginals=marginals, **kwargs)

        return ax

    def _select_random_indices(self, number, total):
        """ Selects a random number (sorted) of indices defined by input number from a total number. If total>=dim returns all. """
        total
        if total<=number:
            indices = np.arange(total)
        else:
            indices = np.random.choice(total, number, replace=False)
            indices.sort()
        return indices

    def to_arviz_inferencedata(self, variable_indices=None):
        """ Return arviz InferenceData object of samples for the given variable indices"""
        # If no variable indices given we convert all
        if variable_indices is None:
            variable_indices = np.arange(self.geometry.dim)

        # Get variable names from geometry
        variables = np.array(self.geometry.variables) #Convert to np array for better slicing
        variables = variables[variable_indices].flatten()

        # Construct inference data structure
        datadict =  dict(zip(variables,self.samples[variable_indices,:]))

        return datadict
        
    def compute_ess(self, **kwargs):
        """ Compute effective sample size (ESS) of samples.
        
        Any remaining keyword arguments will be passed to the arviz computing tool.
        See https://arviz-devs.github.io/arviz/api/generated/arviz.ess.html.

        Returns
        -------
        Numpy array with effective sample size for each variable.
        """
        ESS_xarray = arviz.ess(self.to_arviz_inferencedata(), **kwargs)
        ESS = np.empty(self.geometry.shape)
        for i, (key, value) in enumerate(ESS_xarray.items()):
            ESS[i] = value.to_numpy()
        return ESS

    def compute_rhat(self, chains, **kwargs):
        """ Compute rhat value of samples given list of cuqi.samples.Samples objects (chains) to compare with.
        
        Here rhat values close to 1 indicates the chains have converged to the same distribution.
        
        Parameters
        ----------
        chains : list (or a single Samples object)
            List of cuqi.samples.Samples objects each representing a single MCMC chain to compare with.
            Each Samples object must have the same geometry as the original Samples object.

        Any remaining keyword arguments will be passed to the arviz computing tool.
        See https://arviz-devs.github.io/arviz/api/generated/arviz.rhat.html.

        Returns
        -------
        Numpy array with rhat values for each variable.
        """

        # If single Samples object put into a list
        if isinstance(chains, Samples):
            chains = [chains]

        if not isinstance(chains, list):
            raise TypeError("Chains needs to be a list")

        n_chains = len(chains)
        for i in range(n_chains):
            if self.geometry != chains[i].geometry:
                raise TypeError(f"Geometry of chain {i} does not match Samples geometry.")

        if len(self.samples.shape) != 2:
            raise TypeError("Raw samples within each chain must have len(shape)==2, i.e. (variable, draws) structure.")
        
        # Get variable names from geometry
        variables = np.array(self.geometry.variables) #Convert to np array for better slicing
        variables = variables.flatten()

        # Construct full samples for all chains
        samples = np.empty((self.samples.shape[0], n_chains+1, self.samples.shape[1]))
        samples[:,0,:] = self.samples
        for i, chain in enumerate(chains):
            samples[:,i+1,:] = chain.samples

        # Construct inference data structure
        datadict =  dict(zip(variables,samples))

        # Compute rhat
        RHAT_xarray = arviz.rhat(datadict, **kwargs)

        # Convert to numpy array
        RHAT = np.empty(self.geometry.shape)
        for i, (key, value) in enumerate(RHAT_xarray.items()):
            RHAT[i] = value.to_numpy()
        return RHAT
