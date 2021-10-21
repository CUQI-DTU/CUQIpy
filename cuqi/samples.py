import numpy as np
import matplotlib.pyplot as plt
from cuqi.diagnostics import Geweke
from cuqi.geometry import _DefaultGeometry

class Samples(object):

    def __init__(self, samples, geometry=None):
        self.samples = samples
        self.geometry = geometry

    @property
    def geometry(self):
        if self._geometry is None:
            self._geometry = _DefaultGeometry(grid=np.prod(self.samples.shape[:-1]))
        return self._geometry

    @geometry.setter
    def geometry(self,inGeometry):
        self._geometry = inGeometry

    def burnthin(self, Nb, Nt):
        self.samples = self.samples[Nb::Nt,:]

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
        return plt.plot(self.samples[variable_indices,:].T,*args,**kwargs)

    def plot_ci(self,percent,exact=None,*args,plot_envelope_kwargs={},**kwargs):
        """
        Plots the confidence interval for the samples according to the geometry.

        Parameters
        ---------
        percent : int
            The percent confidence to plot (i.e. 95, 99 etc.)
        
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

        lci = self.geometry.plot_envelope(lo_conf, up_conf,color='dodgerblue',**pe_kwargs)

        lmn = self.geometry.plot(mean,*args,**kwargs)
        if exact is not None: #TODO: Allow exact to be defined in different space than mean?
            lex = self.geometry.plot(exact,*args,**kwargs)
            plt.legend([lmn[0], lex[0], lci],["Mean","Exact","Confidence Interval"])
        else:
            plt.legend([lmn[0], lci],["Mean","Confidence Interval"])

    def diagnostics(self):
        # Geweke test
        Geweke(self.samples.T)
