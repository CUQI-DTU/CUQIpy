# %% Try GMRF distribution in 1D and 2D on a deconvolution problem
import sys; sys.path.append("..")
import cuqi
import numpy as np
import scipy.sparse as sps

# %% Define testproblem
TP = cuqi.testproblem.Deconvolution1D()
TP = cuqi.testproblem.Deconvolution2D() # Uncomment for 2D

# Some helper variables
n = TP.model.domain_dim # Number of parameters
N = TP.model.domain_geometry.fun_shape[0] # Number of parameters in one dimension
ndim = len(TP.model.domain_geometry.fun_shape) # Number of dimensions

# %% Define GMRF prior (structured Gaussian)
prior = cuqi.distribution.GMRF(
    mean=np.zeros(n),
    prec=500, # Higher precision means more regularized solution.
    order=1,  # Higher order mean more correlated structures. Can be 1 or 2.
    bc_type="zero", # Boundary conditions for GMRF. Non-zero are experimental.
    geometry=TP.model.domain_geometry
)

# Plot samples of prior
prior.sample(1).plot()

# %% Set prior and sample posterior
TP.prior = prior
samples = TP.sample_posterior(200)

# Plot samples of posterior
samples.plot_ci(exact=TP.exactSolution)

# %% Using Gaussian instead

P = 500*sps.diags([-1, 2, -1], [-1, 0, 1], shape=(N, N)) # order 1
#P = 500*sps.diags([1, -4, 6, -4, 1], [-2, -1, 0, 1, 2], shape=(N, N)) # order 2

if ndim == 2:
    I = sps.eye(N, dtype=int)
    Ds = sps.kron(I, P)
    Dt = sps.kron(P, I)
    P = Ds+Dt

prior2 = cuqi.distribution.Gaussian(mean=np.zeros(n), prec=P, geometry=TP.model.domain_geometry)

# Plot samples of prior
prior2.sample(5).plot()

# %% Set prior and sample posterior
TP.prior = prior2
samples = TP.sample_posterior(200)

# Plot samples of posterior
samples.plot_ci(exact=TP.exactSolution)

# %%
TP.prior = cuqi.distribution.LMRF(
    location=0,
    scale=0.001,
    geometry=TP.model.domain_geometry
)

samples = TP.sample_posterior(200)

# %%
samples.plot_ci(exact=TP.exactSolution)

# %%
