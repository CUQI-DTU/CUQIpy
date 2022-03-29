# %% Here we test the automatic sampler selection for Deconvolution (1D and 2D).
# TODO:
# 1) Add LMRF_approx?

import sys
sys.path.append("..")
import cuqi
import numpy as np
import matplotlib.pyplot as plt

from cuqi.distribution import Gaussian, GaussianCov, Cauchy_diff, Laplace_diff
from cuqi.distribution import GMRF, LMRF, Laplace, Beta, InverseGamma, Lognormal
from cuqi.sampler import NUTS, CWMH

%load_ext autoreload
%autoreload 2
# %% Deconvolution 1D problem
dim = 128
#TP = cuqi.testproblem.Deconvolution(dim=dim, phantom="square")
TP = cuqi.testproblem.Deconvolution2D(dim=dim, phantom="satellite")
n = TP.model.domain_dim
N = TP.model.domain_geometry.shape[0]
ndim = len(TP.model.domain_geometry.shape)

# Prior par
if ndim == 1: par = 0.02 #1d
if ndim == 2: par = 1 #2d
if ndim == 1: Ns = 1000
if ndim == 2: Ns = 500

# %% Prior choices (Main ones of interest: Gaussian, GMR, Cauchy_diff, Laplace_diff:
# Working choices
#TP.prior = Gaussian(mean=np.zeros(n), std=par, geometry=TP.model.domain_geometry) #Not for 2D.
#TP.prior = GaussianCov(mean=np.zeros(n), cov=par**2, geometry=TP.model.domain_geometry)
#TP.prior = GMRF(np.zeros(n), 1/par**2, N, ndim, "zero", geometry=TP.model.domain_geometry) # Odd behavior (swingy?)

# Seems ok in 1D - (Not really seem feasible in 2D it seems) - Cauchy might work.
TP.prior = Cauchy_diff(location=np.zeros(n), scale=0.01, bc_type="zero", physical_dim=ndim, geometry=TP.model.domain_geometry) #NUTS is not adapting parameters fully. (Not using sample(n,nb) atm.)
#TP.prior = Laplace_diff(location=np.zeros(n), scale=0.01, bc_type="zero", physical_dim=ndim, geometry=TP.model.domain_geometry) #Does not veer away from initial guess very much in prior samples
#TP.prior = LMRF(np.zeros(n), 1/par**2, N, ndim, "zero", geometry=TP.model.domain_geometry) # Seems OK. Same as Laplace diff actually?

# Bad choices (ignore) both 1D and 2D
#TP.prior = Beta(2*np.ones(n), 5*np.ones(n)) #Might need tuning
#TP.prior = Laplace(np.zeros(n), 10) #Might need tuning
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
    plt.plot(TP.exactSolution, label="exact")
    plt.legend()
elif ndim == 2:
    plt.figure()
    cuqi.samples.CUQIarray(x_map, geometry=TP.model.domain_geometry).plot()
    plt.figure()
    plt.subplot(122); TP.exactSolution.plot()

# %% Prior samples
try:
    print("Trying direct..")
    samples_prior = TP.prior.sample(3)
    
except:
    try:
        print("Trying NUTS..")
        samples_prior = NUTS(TP.prior).sample(Ns)
    except:
        print("Trying CWMH..")
        samples_prior = CWMH(TP.prior).sample_adapt(Ns)
samples_prior.plot()
# %%
