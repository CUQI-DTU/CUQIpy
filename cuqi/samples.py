import numpy as np
import matplotlib.pyplot as plt

class Samples(object):

    def __init__(self, samples):
        self.samples = samples
    
    def burnthin(self, Nb, Nt):
        self.samples = self.samples[Nb::Nt,:]

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
