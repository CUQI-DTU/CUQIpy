import numpy as np
import matplotlib.pyplot as plt
from cuqi.diagnostics import Geweke
from cuqi.geometry import _DefaultGeometry1D, Continuous2D, Image2D
from cuqi.array import CUQIarray
from cuqi.utilities import force_ndarray
from copy import copy
from numbers import Number

try:
    import arviz  # Plotting tool
except ImportError as e:
    arviz = None
    arviz_import_error = e


def _check_for_arviz():
    if arviz is None:
        msg = "The arviz package is required for this functionality. "\
            + "Please make sure arviz is installed. "\
            + "See below for the original error message:\n"\
            + arviz_import_error.args[0]

        raise ImportError(msg)


class Samples(object):
    """
    An object used to store samples from distributions. 

    Parameters
    ----------
    samples : ndarray
        Contains the raw samples as a numpy array indexed by the last axis of the array.

    geometry : cuqi.geometry.Geometry, default None
        Contains the geometry related of the samples

    """
    def __init__(self, samples, geometry=None, is_par=True, is_vec=True):
        self.geometry = geometry
        self.is_par = is_par
        self.samples = samples
        self.is_vec = is_vec

    def _sub_samples(self, indices):
        """Returns a new Samples object with the samples indexed by indices."""
        sub_samples = self.samples[..., indices]
        if isinstance(indices, Number):
            sub_samples = sub_samples[..., np.newaxis]

        return Samples(sub_samples,
                       geometry=self.geometry,
                       is_par=self.is_par,
                       is_vec=self.is_vec)

    def __iter__(self):
        """Returns an iterator over the samples"""
        if isinstance(self.samples, list):
            for i in range(self.Ns):
                yield self.samples[i]
        else:
            for i in range(self.Ns):
                yield self.samples[..., i]

    @property
    def shape(self):
        """Returns the shape of samples."""
        return self.samples.shape

    @property
    def Ns(self):
        """Return number of samples"""
        if isinstance(self.samples, list):
            return len(self.samples)
        else:
            return self.samples.shape[-1]

    @property
    def geometry(self):
        if self._geometry is None:
            self._geometry = _DefaultGeometry1D(grid=np.prod(self.samples.shape[:-1]))
        return self._geometry

    @property
    def is_vec(self):
        return self._is_vec
    
    @is_vec.setter
    def is_vec(self, value):
        if self.is_par and not value:
            raise ValueError("Cannot set is_vec to False when is_par is True")
        self._is_vec = value

    @property
    def funvals(self):
        """Returns a new Samples object of sample function values. If samples are already function values, the Samples object itself is returned.""" 

        # Return self if the samples are function values
        # i.e. not parameters or vector representations
        # of function values
        if not self.is_par and not self.is_vec:
            return self
        
        # Set conversion method either par2fun or vec2fun
        if self.is_par:
            convert = self.geometry.par2fun
        else:
            convert = self.geometry.vec2fun

        # Convert the samples to function values
        
        # If the function representation is an array, return funvals samples
        # as an array, else, return a list of function values 
        if self.geometry.fun_is_array:
            funvals = np.empty(self.geometry.fun_shape+(self.Ns,))
            for i, value in enumerate(self):
                funvals[..., i] = convert(value)
        else:
            funvals = [convert(value) for value in self]

        # Check if the function values are in vector representations after
        # conversion
        if isinstance(funvals, np.ndarray) and len(funvals.shape) <= 2:
            is_vec = True 
        else:
            is_vec = False

        # Create and return a new Samples object of function values
        return Samples(funvals, is_par=False, is_vec=is_vec,
                       geometry=self.geometry)
            
    @property
    def vector(self):
        """Returns a new Samples object of samples in vector form. If samples are already in vector form (e.g. samples of parameter values, or samples of function values that are already converted to vector format), the Samples object itself is returned."""

        # Return self if the samples are in vector form
        if self.is_vec or self.is_par:
            return self

        # In the remaining case, the samples are function values
        # Set conversion method to fun2vec 
        convert = self.geometry.fun2vec

        # Convert the samples to vector form
        vecvals = np.empty((self.geometry.funvec_dim, self.Ns))
        for i, value in enumerate(self):
            vecvals[..., i] = convert(value)

        # Create and return a new Samples object of vector samples
        return Samples(vecvals, is_par=self.is_par, is_vec=True,
                       geometry=self.geometry)

    @property
    def parameters(self):
        """If `self.is_par` is False, returns a new Samples object of sample parameters by converting the function values to parameters. If `self.is_par` is True, returns the Samples object itself."""

        # Return self if the samples are parameters
        if self.is_par:
            return self
        
        # Set conversion method fun2par or vec2fun(fun2par(...))
        if not self.is_vec:
            convert = self.geometry.fun2par
        else:
            convert = \
                lambda vec: self.geometry.fun2par(self.geometry.vec2fun(vec))
            
        # Convert the samples to parameter values
        parameters = np.empty((self.geometry.par_dim, self.Ns))
        for i, value in enumerate(self):
            parameters[:, i] = convert(value)

        # Create and return a new Samples object of parameters
        return Samples(parameters, is_par=True, is_vec=True,
                       geometry=self.geometry)

    @property
    def _geometry_dim(self):
        if self.is_par: # if parameters
            return self.geometry.par_dim
        elif self.is_vec: # if not parameters but vector representation of function
            return self.geometry.funvec_dim
        else: # if function values
            return self.geometry.fun_dim
        
    @property
    def _geometry_shape(self):
        if self.is_par: # if parameters
            return self.geometry.par_shape
        elif self.is_vec: # if not parameters but vector representation of function
            return self.geometry.funvec_shape
        else: # if function values
            return self.geometry.fun_shape

    @geometry.setter
    def geometry(self,inGeometry):
        self._geometry = inGeometry

    def _process_is_par_kwarg(self, plotting_dict):
        """Updates the plotting dictionary by setting the is_par attribute
        of plotting_dict to the value of self.is_par"""
        if "is_par" in plotting_dict.keys():
            raise ValueError(
                "Cannot pass is_par as a plotting argument. "+
                "is_par is determined automatically by the samples object.")
        plotting_dict["is_par"] = self.is_par

    def _convert_to_funvals_if_needed(self, value):
        """Converts the input value to function values if the value is a vector 
        representation of function values"""
        if not self.is_par and self.is_vec:
            return self.geometry.vec2fun(value)
        else:
            return value
        
    def _compute_numpy_stats(self, method, *args, **kwargs):
        """Apply the numpy method `method` to `self.samples`. Additional 
        arguments `args` and keyword arguments `kwargs` are passed to the numpy
        method `method`."""

        # Compute the statistics and catch TypeError if the samples does
        # not have the correct data type
        try:
            stats = method(self.samples, *args, **kwargs)

        except Exception as e:
            msg = f"{self.__module__} added message: Cannot compute statistics for the given samples. Only numpy arrays are supported. Consider using the property vector to convert the samples to vector representation, e.g. my_samples.vector, to be able to compute statistics on the samples. See below for the original error message:\n"
            e.args = (msg + e.args[0],) + e.args[1:]
            raise e
   
        return stats
    
    def _raise_error_if_not_vec(self, method_name):
        """Raises an error if the samples are not in vector form when calling a
        method that requires the samples to be in vector form."""

        # Raise an error if the samples are not in vector form
        if not self.is_vec:
            raise ValueError(
                "Cannot perform "+method_name+" on samples that are not in "+
                "vector form. Consider using the property vector to convert "+
                "the samples to vector representation, e.g. my_samples.vector, "+
                "before calling "+method_name+".")

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

    def plot_mean(self, *args, **kwargs):
        """Plot pointwise mean of the samples

        Positional and keyword arguments are passed to the underlying `self.geometry.plot` method.
        See documentation of `self.geometry` for options.
        """
        self._process_is_par_kwarg(kwargs)

        mean = self.mean()
        
        # If mean is function in vector form, convert to function values
        mean = self._convert_to_funvals_if_needed(mean)

        # Plot mean according to geometry
        ax =  self.geometry.plot(mean, *args, **kwargs)
        plt.title('Sample mean')
        return ax

    def plot_median(self,*args,**kwargs):
        """Plot pointwise median of the samples

        Positional and keyword arguments are passed to the underlying `self.geometry.plot` method.
        See documentation of `self.geometry` for options.
        """
        self._process_is_par_kwarg(kwargs)

        median = self.median()

        # If median is function in vector form, convert to function values
        median = self._convert_to_funvals_if_needed(median)

        # Plot median according to geometry
        ax =  self.geometry.plot(median, *args, **kwargs)
        plt.title('Pointwise sample median')
        return ax

    def plot_variance(self, *args, **kwargs):
        """Plot pointwise variance of the samples

        Positional and keyword arguments are passed to the underlying `self.geometry.plot` method.
        See documentation of `self.geometry` for options.
        """
        self._process_is_par_kwarg(kwargs)

        variance = self.variance()

        # If variance is function in vector form, convert to function values
        variance = self._convert_to_funvals_if_needed(variance)

        # Plot variance according to geometry
        ax = self.geometry.plot(variance, *args, **kwargs)
        plt.title('Sample variance')
        return ax

    def plot_ci_width(self,percent=95,*args,**kwargs):
        """Plot width of the pointwise credibility intervals of the samples

        Positional and keyword arguments are passed to the underlying `self.geometry.plot` method.
        See documentation of `self.geometry` for options.
        """
        self._process_is_par_kwarg(kwargs)

        ci_width = self.ci_width(percent)

        # If ci_width is function in vector form, convert to function values
        ci_width = self._convert_to_funvals_if_needed(ci_width)

        # Plot width of credibility intervals according to geometry
        ax = self.geometry.plot(ci_width, *args, **kwargs)
        plt.title('Width of sample credibility intervals')
        return ax

    def mean(self):
        """Compute mean of the samples."""
        return self._compute_numpy_stats(np.mean, axis=-1)

    def median(self):
        """Compute pointwise median of the samples"""
        return self._compute_numpy_stats(np.median, axis=-1)

    def variance(self):
        """Compute pointwise variance of the samples"""
        return self._compute_numpy_stats(np.var, axis=-1)

    def compute_ci(self, percent=95):
        """Compute pointwise credibility intervals of the samples."""
        lb = (100-percent)/2
        up = 100-lb
        return self._compute_numpy_stats(
            np.percentile, [lb, up], axis=-1) 

    def ci_width(self, percent = 95):
        """Compute width of the pointwise credibility intervals of the samples"""
        lo_conf, up_conf = self.compute_ci(percent)
        return up_conf-lo_conf
    
    def std(self):
        """Compute pointwise standard deviation of the samples"""
        return self._compute_numpy_stats(np.std, axis=-1)

    def plot_std(self,*args,**kwargs):
        """Plot pointwise standard deviation of the samples

        Positional and keyword arguments are passed to the underlying `self.geometry.plot` method.
        See documentation of `self.geometry` for options.
        """
        self._process_is_par_kwarg(kwargs)
        # Compute std assuming samples are index in last dimension of nparray
        std = self.std()
        
        # If std is function in vector form, convert to function values
        std = self._convert_to_funvals_if_needed(std)

        # Plot std according to geometry
        ax = self.geometry.plot(std, *args, **kwargs)
        plt.title('Sample standard deviation')
        return ax

    def plot(self,sample_indices=None,*args,**kwargs):
        """ Plots one or more samples. """
        Ns = self.Ns
        Np = 5 # Number of samples to plot if Ns > 5

        self._process_is_par_kwarg(kwargs)

        if sample_indices is None:
            if Ns>Np: print("Plotting {} randomly selected samples".format(Np))
            sample_indices = self._select_random_indices(Np, Ns)
        plot_samples = self._sub_samples(sample_indices)

        # If samples are function values in vector form, convert to function
        # values
        if not self.is_par and self.is_vec:
            plot_samples = plot_samples.funvals.samples
        else:
            plot_samples = plot_samples.samples

        # Plot samples according to geometry
        return self.geometry.plot(plot_samples, *args, **kwargs)
        
    def plot_chain(self, variable_indices=None, *args, **kwargs):

        self._raise_error_if_not_vec(self.plot_chain.__name__)

        dim = self._geometry_dim
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
        """ Plots samples histogram of variables with indices specified in variable_indices. """

        self._raise_error_if_not_vec(self.hist_chain.__name__)

        if 'label' in kwargs.keys():
            raise Exception("Argument 'label' cannot be passed by the user")
        variables = np.array(self.geometry.variables) #Convert to np array for better slicing
        variables = variables[variable_indices].flatten()
        n, bins, patches = plt.hist(self.samples[variable_indices,:].T,*args,**kwargs)
        plt.legend(variables)
        return patches

    def plot_ci(self, percent=95, exact=None, *args, plot_envelope_kwargs=None, **kwargs):
        """
        Plots the credibility interval for the samples according to the geometry.

        Parameters
        ----------
        percent : int
            The percent credibility to plot (i.e. 95, 99 etc.)
        
        exact : ndarray, default None
            The exact value (for comparison)
        plot_envelope_kwargs : dict, default {}
            Keyword arguments for the plot_envelope method

        Returns
        -------
        plotting_objects : list
            If 1D plots are generated, the list contains 
            :class:`~matplotlib.lines.Line2D` object of the mean plot, 
            :class:`~matplotlib.lines.Line2D` object of the exact value plot,
            and :class:`~matplotlib.collections.PolyCollection`
            or :class:`~matplotlib.container.ErrorbarContainer`
            object of the ci envelope plot, respectively.

            If 2D plots are generated, the list contains
            :class:`~matplotlib.collections.PolyCollection` or 
            :class:`~matplotlib.image.AxesImage` objects
            for the mean, exact value, ci lower bound, ci upper
            bound, and the ci width, respectively.
        """
        
        # Compute statistics
        lo_conf, up_conf = self.compute_ci(percent)

        #Extract plotting keywords and put into plot_envelope
        if plot_envelope_kwargs is None:
            plot_envelope_kwargs = {}
        pe_kwargs = plot_envelope_kwargs

        # is_par is determined automatically from self.is_par 
        # Depending on the value of self.is_par, the computed statistics below
        # (mean, lo_conf,up_conf) are either parameter values or function values
        self._process_is_par_kwarg(kwargs)
        self._process_is_par_kwarg(pe_kwargs)

        # Create a copy of kwargs without is_par
        kwargs_copy = kwargs.copy()
        kwargs_copy.pop("is_par")

        #User cannot ask for computing statistics on function values then plotting on parameter space
        if not self.is_par:
            if "plot_par" in kwargs and kwargs["plot_par"] or\
                    "plot_par" in pe_kwargs and pe_kwargs["plot_par"]:
                #TODO: could be allowed if the underlying plotting functions will convert the samples to parameter space
                raise ValueError(
                    "Cannot plot credible interval on parameter space if the samples are in the function space.")

        # Set plot_par value to be passed to Geometry.plot_envelope and Geometry.plot.
        if "plot_par" in kwargs:
            pe_kwargs["plot_par"] = kwargs["plot_par"]
        else:
            pe_kwargs["plot_par"] = False
            kwargs["plot_par"] = False

        if (type(self.geometry) is Continuous2D or type(self.geometry) is Image2D) and not kwargs["plot_par"]:

            # Create variables for returned values from geometry plotting methods
            im_mn, im_ex, im_lo, im_up, im_wd = None, None, None, None, None

            plt.figure()
            #fig.add_subplot(2,2,1)
            im_mn = self.plot_mean(*args, **kwargs_copy)
            plt.title("Sample mean")
            if exact is not None:
                #fig.add_subplot(2,2,3)
                plt.figure()
                im_ex = self.geometry.plot(exact, *args, **kwargs)
                plt.title("Exact")
            #fig.add_subplot(2,2,2)
            plt.figure()
            im_wd = self.geometry.plot(up_conf-lo_conf)
            plt.title("Width of credibility interval")
            plt.figure()
            im_up = self.geometry.plot(up_conf)
            plt.title("Upper credibility interval limit")
            #fig.add_subplot(2,2,4)
            plt.figure()
            im_lo = self.geometry.plot(lo_conf)
            plt.title("Lower credibility interval limit")

            plotting_objects = [im_mn, im_ex, im_lo, im_up, im_wd]
        else:
            # Create variables for returned values from geometry plotting methods
            lci, lmn, lex = None, None, None

            lci = self.geometry.plot_envelope(
                lo_conf, up_conf,
                color='dodgerblue',
                **pe_kwargs,
                label=f"{percent}% Credibility Interval")

            lmn = self.plot_mean(*args, **kwargs_copy, label="Mean")
            plt.title("")
            if exact is not None:
            #TODO: Allow exact to be defined in different space than mean?
                if isinstance(exact, CUQIarray):
                    lex = exact.plot(*args, **kwargs, label="Exact")
                else:
                    lex = self.geometry.plot(exact, *args, **kwargs, label="Exact")
            plt.legend()

            plotting_objects = [lmn, lex, lci]
        
        # Form a list of the matplotlib objects that were plotted
        # Note that in 1D case, `self.geometry.plot` returns a list of
        # one object that we need to extract, hence we use indexing: obj[0].
        plotting_objects = [obj[0] if (type(obj) is list and obj is not None)
                            else obj for obj in plotting_objects]
        return plotting_objects


    def diagnostics(self):
        """ Conducts diagnostics on the chain (Geweke test). """
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
        dim = self._geometry_dim
        Nv = 5 # Max number of variables to plot if none are chosen

        # If no variables are given we randomly select some at random
        if variable_indices is None:
            if Nv<dim: print("Selecting 5 randomly chosen variables")
            variable_indices = self._select_random_indices(Nv, dim)

        # Convert to arviz InferenceData object
        datadict = self.to_arviz_inferencedata(variable_indices)
        
        # Plot autocorrelation using arviz
        _check_for_arviz()
        axis = arviz.plot_autocorr(datadict, max_lag=max_lag, combined=combined, **kwargs)

        return axis

    def plot_trace(self, variable_indices=None, exact=None, combined=True, tight_layout=True, **kwargs):
        """Creates a traceplot of the samples consisting of 1) a histogram/density plot of the samples and 2) an MCMC chain plot.
        
        Parameters
        ----------
        variable_indices : list, optional
            List of variable indices to plot the autocorrelation for. If no input is given and less than 5 variables exist all are plotted and with more 5 are randomly chosen.

        exact : array-like, optional
            Exact solution to compare with the samples.

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
        dim = self._geometry_dim
        Nv = 5 # Max number of variables to plot if none are chosen

        # If no variables are given we randomly select some at random
        if variable_indices is None:
            if Nv<dim: print("Selecting 5 randomly chosen variables")
            variable_indices = self._select_random_indices(Nv, dim)

        # Convert to arviz InferenceData object
        datadict = self.to_arviz_inferencedata(variable_indices)

        # If exact solution is given, add it via lines argument to arviz using the variable names
        if exact is not None:

            # Convert exact to ndarray (in case single parameter etc.)
            exact = force_ndarray(exact, flatten=True)

            # Attempt to extract variables of the exact solution if given in full dimension
            if len(exact) == dim:
                exact = exact[variable_indices]

            # If exact is given in reduced dimension, check that the number of variables match
            if len(variable_indices) != len(exact):
                raise ValueError(f"The shape of the exact argument {exact.shape} must match the number of variables to be plotted by variable indices {variable_indices.shape}.")
            
            # Extract variable names
            par_names = [self.geometry.variables[i] for i in variable_indices]

            # Arviz style adding lines to traceplot
            if "lines" in kwargs:
                raise ValueError("The lines argument is already defined in kwargs. Please remove it to use the exact keyword argument.")
            
            kwargs["lines"] = tuple([(par_names[i], {}, exact[i]) for i in range(len(par_names))])

        # Plot using arviz
        _check_for_arviz()
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
        dim = self._geometry_dim
        Nv = 5 # Max number of variables to plot if none are chosen

        # If no variables are given we randomly select some at random
        if variable_indices is None:
            if Nv<dim: print("Selecting 5 randomly chosen variables")
            variable_indices = self._select_random_indices(Nv, dim)

        # Convert to arviz InferenceData object
        datadict = self.to_arviz_inferencedata(variable_indices)

        _check_for_arviz()
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
        # If samples are not in a vector representation, i.e. the samples is
        # not a 2D numpy array, we cannot convert to arviz InferenceData object
        if not self.is_vec:
            raise ValueError("Samples are not in a vector representation. "+ 
                "Cannot convert to arviz InferenceData object. Consider using "+
                "the `Samples` property `vector` to convert to vector "+
                "representation.")

        # If no variable indices given we convert all
        if variable_indices is None:
            variable_indices = np.arange(self._geometry_dim)

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
        _check_for_arviz()
        ESS_xarray = arviz.ess(self.to_arviz_inferencedata(), **kwargs)
        ESS_items = ESS_xarray.items()
        ESS = np.empty(len(ESS_items))
        for i, (key, value) in enumerate(ESS_items):
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
        _check_for_arviz()
        RHAT_xarray = arviz.rhat(datadict, **kwargs)

        # Convert to numpy array
        RHAT = np.empty(self._geometry_shape)
        for i, (key, value) in enumerate(RHAT_xarray.items()):
            RHAT[i] = value.to_numpy()
        return RHAT

    def plot_violin(self, variable_indices=None, **kwargs):
        """ Create a violin plot of the samples. 
        
        Parameters
        ----------
        variable_indices : list, optional
            List of variable indices to plot.
            If no input is given and less than 8 variables exist all
            are plotted and with more 8 are randomly chosen.

        Any remaining keyword arguments will be passed to the arviz plotting tool.
        See https://arviz-devs.github.io/arviz/api/generated/arviz.plot_violin.html.

        Returns
        -------
        axes: matplotlib axes or bokeh figures
        
        """
        dim = self._geometry_dim
        Nv = 8 # Max number of variables to plot if none are chosen

        # If no variables are given we randomly select some at random
        if variable_indices is None:
            if Nv<dim: print(f"Selecting {Nv} randomly chosen variables")
            variable_indices = self._select_random_indices(Nv, dim)

        # Convert to arviz InferenceData object
        datadict = self.to_arviz_inferencedata(variable_indices)

        # Plot using arviz
        _check_for_arviz()
        ax =  arviz.plot_violin(datadict, **kwargs)

        return ax

    def __repr__(self) -> str: 
        return "CUQIpy Samples:\n" + \
               "---------------\n\n" + \
               "Ns (number of samples):\n {}\n\n".format(self.Ns) + \
               "Geometry:\n {}\n\n".format(self.geometry) + \
               "Shape:\n {}\n\n".format(self.shape) + \
               "Samples:\n {}\n\n".format(self.samples)

class JointSamples(dict):
    """ An object used to store samples from :class:`cuqi.distribution.JointDistribution`. 

    This object is a simple overload of the dictionary class to allow easy access to certain methods 
    of Samples objects without having to iterate over each key in the dictionary. 
    
    """

    def burnthin(self, Nb, Nt=1):
        """ Remove burn-in and thin samples for all samples in the dictionary. Returns a copy of the samples stored in the dictionary. """
        return JointSamples({key: samples.burnthin(Nb, Nt) for key, samples in self.items()})

    def __repr__(self) -> str: 
        return "CUQIpy JointSamples Dict:\n" + \
               "-------------------------\n\n" + \
               "Keys:\n {}\n\n".format(list(self.keys())) + \
               "Ns (number of samples):\n {}\n\n".format({key: samples.Ns for key, samples in self.items()}) + \
               "Geometry:\n {}\n\n".format({key: samples.geometry for key, samples in self.items()}) + \
               "Shape:\n {}\n\n".format({key: samples.shape for key, samples in self.items()})
