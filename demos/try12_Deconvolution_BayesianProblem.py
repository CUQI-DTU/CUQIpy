# %% Here we test the automatic sampler selection for Deconvolution (1D and 2D).
# This script can be explored by commenting out various priors for either the 1D or 2D deconvolution problem.
import sys
sys.path.append("..")
import cuqi
import numpy as np
import matplotlib.pyplot as plt

from cuqi.distribution import Gaussian, CMRF, LMRF
from cuqi.distribution import GMRF, LMRF, Laplace, Beta, InverseGamma, Lognormal
from cuqi.sampler import NUTS, CWMH

# %% Deconvolution 1D problem
dim = 128
#TP = cuqi.testproblem.Deconvolution1D(dim=dim, phantom="square")
TP = cuqi.testproblem.Deconvolution2D(dim=dim, phantom=cuqi.data.grains(size=dim))
n = TP.model.domain_dim
N = TP.model.domain_geometry.fun_shape[0]
ndim = len(TP.model.domain_geometry.fun_shape)

# Prior par
if ndim == 1: par = 0.02 #1d
if ndim == 2: par = 1 #2d
if ndim == 1: Ns = 1000
if ndim == 2: Ns = 500

# %% Prior choices (Main ones of interest: Gaussian, GMR, CMRF, LMRF:
# Working choices
#TP.prior = Gaussian(mean=np.zeros(n), cov=par**2, geometry=TP.model.domain_geometry)
#TP.prior = GMRF(np.zeros(n), 1/par**2, "zero", geometry=TP.model.domain_geometry) # Odd behavior (swingy?)

TP.prior = CMRF(location=0, scale=0.01, bc_type="zero", geometry=TP.model.domain_geometry)
#TP.prior = LMRF(location=0, scale=0.01, bc_type="neumann", geometry=TP.model.domain_geometry)

# Bad choices (ignore) both 1D and 2D
#TP.prior = Beta(2*np.ones(n), 5*np.ones(n)) #Might need tuning
#TP.prior = Laplace(np.zeros(n), 1/10) #Might need tuning
#TP.prior = InverseGamma(3*np.ones(n), np.zeros(n), 1*np.ones(n)) #Bad choice in general.. #Might need tuning
#TP.prior = Lognormal(mean=np.zeros(n), cov=0.05) #NUTS ACTS out!

# %% Samples
samples = TP.sample_posterior(Ns)

# %% CI plot
samples.plot_ci(exact=TP.exactSolution)

# %% plot
samples.plot()

# %% traceplot
samples.plot_trace()

# %% autocorr
samples.plot_autocorrelation()

# %% scatter
samples.plot_pair()

# %% MAP ESTIMATE
x_map, info = TP.MAP()

if ndim == 1:
    plt.plot(x_map, label="MAP")
    TP.exactSolution.plot(label="exact")
    plt.legend()
elif ndim == 2:
    plt.figure()
    cuqi.array.CUQIarray(x_map, geometry=TP.model.domain_geometry).plot()
    plt.figure()
    plt.subplot(122); TP.exactSolution.plot()

# %% Prior samples (this is a bit messy)
try:
    print("Trying direct..")
    samples_prior = TP.prior.sample(3)
except NotImplementedError:
    try:
        print("Trying NUTS..")
        samples_prior = NUTS(TP.prior).sample(Ns)
    except (NotImplementedError, AttributeError): #Some methods do not have "gradient" so attribute error also
        print("Trying CWMH..")
        samples_prior = CWMH(TP.prior).sample_adapt(Ns)
samples_prior.plot()
# %%
