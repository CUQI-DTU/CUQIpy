import numpy as np
import matplotlib.pyplot as plt
from cuqi.diagnostics import Geweke
from cuqi.geometry import Continous1D

class Samples(object):

    def __init__(self, samples, geometry=None):
        self.samples = samples
        self.geometry = geometry

    @property
    def geometry(self):
        if self._geometry is None:
            self._geometry = Continous1D(dim=self.samples.shape[:-1])
        return self._geometry

    @geometry.setter
    def geometry(self,inGeometry):
        self._geometry = inGeometry

    def burnthin(self, Nb, Nt):
        self.samples = self.samples[Nb::Nt,:]

    def plot_mean(self):
        # Compute mean assuming samples are index in last dimension of nparray
        mean = np.mean(self.samples,axis=-1)

        # Plot mean according to geometry
        return self.geometry.plot(mean)

        # Potentially return figure handle?

    def plot(self,sample_indicies=None):
        Ns = self.samples.shape[-1]
        if sample_indicies is None:
            if Ns < 10:
                return self.geometry.plot(self.samples)
            else:
                print("Plotting 5 randomly selected samples")
                return self.geometry.plot(self.samples[:,np.random.choice(Ns,5,replace=False)])
        else:
            return self.geometry.plot(self.samples[:,sample_indicies])

    def plot_chain(self,parameter_indicies):
        return plt.plot(self.samples[parameter_indicies,:])

    def plot_ci(self,percent,exact=None):

        if not isinstance(self.geometry,Continous1D):
            raise NotImplementedError("Confidence interval not implemented for {}".format(self.geometry))
        
        # Compute statistics
        mean = np.mean(self.samples,axis=-1)
        lb = (100-percent)/2
        up = 100-lb
        lo_conf, up_conf = np.percentile(self.samples, [lb, up], axis=-1)

        #Plot
        if exact is not None:
            self.geometry.plot(exact,'.-')
        self.geometry.plot(mean,'.-')
        plt.fill_between(self.geometry.grid,up_conf, lo_conf, color='dodgerblue', alpha=0.25)
        if exact is not None:
            plt.legend(["Exact","Mean","Confidence Interval"])
        else:
            plt.legend(["Mean","Confidence Interval"])

    def diagnostics(self):
        # Geweke test
        Geweke(self.samples.T)
