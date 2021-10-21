import numpy as np
import matplotlib.pyplot as plt
from cuqi.diagnostics import Geweke
from cuqi.geometry import Continuous1D, Discrete, _DefaultGeometry
from copy import copy

class Samples(object):
    """
    An object used to store samples from distributions. 

    Parameters
    ----------
    samples : ndarray
        Contains the raw samples as a numpy array indexed by the last axis of the array.

    geometry : cuqi.geometry.Geometry, default None
        Contains the geometry related of the samples

    Methods
    ----------
    :meth:`plot`: Plots one or more samples.
    :meth:`plot_ci`: Plots a confidence interval for the samples.
    :meth:`plot_mean`: Plots the mean of the samples.
    :meth:`plot_std`: Plots the std of the samples.
    :meth:`plot_chain`: Plots all samples of one or more variables (MCMC chain).
    :meth:`burnthin`: Removes burn-in and thins samples.
    :meth:`diagnostics`: Conducts diagnostics on the chain.
    :meth:`shape`: Returns the shape of samples.
    """
    def __init__(self, samples, geometry=None):
        self.samples = samples
        self.geometry = geometry

    @property
    def shape(self):
        return self.samples.shape

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
        new_samples = copy(self)
        new_samples.samples = self.samples[...,Nb::Nt]
        return new_samples

    def plot_mean(self,*args,**kwargs):
        # Compute mean assuming samples are index in last dimension of nparray
        mean = np.mean(self.samples,axis=-1)

        # Plot mean according to geometry
        return self.geometry.plot(mean,*args,**kwargs)

    def plot_std(self,*args,**kwargs):
        # Compute std assuming samples are index in last dimension of nparray
        std = np.std(self.samples,axis=-1)

        # Plot mean according to geometry
        return self.geometry.plot(std,*args,**kwargs)

    def plot(self,sample_indices=None,*args,**kwargs):
        Ns = self.samples.shape[-1]
        if sample_indices is None:
            if Ns < 10:
                return self.geometry.plot(self.samples,*args,**kwargs)
            else:
                print("Plotting 5 randomly selected samples")
                return self.geometry.plot(self.samples[:,np.random.choice(Ns,5,replace=False)],*args,**kwargs)
        else:
            return self.geometry.plot(self.samples[:,sample_indices],*args,**kwargs)

    def plot_chain(self,variable_indices,*args,**kwargs):
        if 'label' in kwargs.keys():
            raise Exception("Argument 'label' cannot be passed by the user")
        if hasattr(self.geometry,"variables"):
            variables = np.array(self.geometry.variables) #Convert to np array for better slicing
            variables = np.array(variables[variable_indices]).flatten()
        else:
            variables = np.array(variable_indices).flatten()
        lines = plt.plot(self.samples[variable_indices,:].T,*args,**kwargs)
        plt.legend(variables)
        return lines

    def plot_ci(self,percent,exact=None,*args,**kwargs):

        if not isinstance(self.geometry,(Continuous1D,Discrete)):
            raise NotImplementedError("Confidence interval not implemented for {}".format(self.geometry))
        
        # Compute statistics
        mean = np.mean(self.samples,axis=-1)
        lb = (100-percent)/2
        up = 100-lb
        lo_conf, up_conf = np.percentile(self.samples, [lb, up], axis=-1)

        lci = self.geometry.plot_envelope(lo_conf, up_conf, color='dodgerblue')

        lmn = self.geometry.plot(mean,*args,**kwargs)
        if exact is not None:
            lex = self.geometry.plot(exact,*args,**kwargs)
            plt.legend([lmn[0], lex[0], lci],["Mean","Exact","Confidence Interval"])
        else:
            plt.legend([lmn[0], lci],["Mean","Confidence Interval"])

    def diagnostics(self):
        # Geweke test
        Geweke(self.samples.T)
