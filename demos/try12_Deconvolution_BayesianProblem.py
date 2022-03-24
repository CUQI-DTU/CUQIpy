# %% Here we test the automatic sampler selection for Deconvolution (1D and 2D).
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
TP = cuqi.testproblem.Deconvolution()
#TP = cuqi.testproblem.Deconvolution2D()
n = TP.model.domain_dim
N = TP.model.domain_geometry.shape[0]
ndim = len(TP.model.domain_geometry.shape)

# Prior par
if ndim == 1: par = 0.05 #1d
if ndim == 2: par = 1 #2d
if ndim == 1: Ns = 1000
if ndim == 2: Ns = 100

# %% Prior choices (Main ones of interest: Gaussian, GMR, Cauchy_diff, Laplace_diff:
# Working choices
#TP.prior = Gaussian(mean=np.zeros(n), std=par, geometry=TP.model.domain_geometry) #Not for 2D.
#TP.prior = GaussianCov(mean=np.zeros(n), cov=par**2, geometry=TP.model.domain_geometry)

# Seems ok? - (Needs fix in 2D)
TP.prior = Cauchy_diff(location=np.zeros(n), scale=par, bc_type="zero", geometry=TP.model.domain_geometry) #NUTS is not adapting parameters fully. (Not using sample(n,nb) atm.)
#TP.prior = Laplace_diff(location=np.zeros(n), scale=par, bc_type="zero", geometry=TP.model.domain_geometry) #Does not veer away from initial guess very much in prior samples

# Need tuning - (Needs fix in 2D)
#TP.prior = GMRF(np.zeros(n), 1/par**2, N, ndim, "zero", geometry=TP.model.domain_geometry) # Odd behavior (swingy?)
#TP.prior = LMRF(np.zeros(n), 1/par**2, N, ndim, "zero", geometry=TP.model.domain_geometry) # Seems OK. Same as Laplace diff actually?

# Bad choices (ignore)
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

# %% Prior samples
try:
    samples_prior = TP.prior.sample(3)
    print("Direct")
except:
    try:
        samples_prior = NUTS(TP.prior).sample(1000)
        print("NUTS")
    except:
        samples_prior = CWMH(TP.prior).sample_adapt(5000)
        print("CWMH")
samples_prior.plot()
# %%
