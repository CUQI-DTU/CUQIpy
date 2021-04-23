import numpy as np
import matplotlib.pyplot as plt
from cuqi.diagnostics import Geweke

class Samples(object):

    def __init__(self, samples):
        self.samples = samples
    
    def burnthin(self, Nb, Nt):
        self.samples = self.samples[Nb::Nt,:]

    def plot(self):
        Ns = self.samples.shape[-1]
        if Ns < 10:
            plt.plot(self.samples)
        else:
            print("Plotting 10 randomly selected samples")
            plt.plot(self.samples[:,np.random.choice(Ns,5,replace=False)])

    def plot_chain(self,index):
        plt.plot(self.samples[index,:])

    def plot_ci(self,percent,exact=None):
        if exact is not None:
            plt.plot(exact,'.-')

        dim = self.samples.shape
        x_mean = np.mean(self.samples,axis=1)
        lb = (100-percent)/2
        up = 100-lb
        x_lo95, x_up95 = np.percentile(self.samples, [lb, up], axis=1)
        plt.plot(x_mean,'.-')
        plt.fill_between(np.arange(dim[0]),x_up95, x_lo95, color='dodgerblue', alpha=0.25)
        if exact is not None:
            plt.legend(["Exact","Mean","Confidence Interval"])
        else:
            plt.legend(["Mean","Confidence Interval"])

    def diagnostics(self):
        # Geweke test
        Geweke(self.samples.T)
