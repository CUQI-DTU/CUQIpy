# %% Try GMRF distribution in 1D and 2D on a deconvolution problem
import sys; sys.path.append("..")
import cuqi
import numpy as np

# %% Define testproblem
TP = cuqi.testproblem.Deconvolution1D()
#TP = cuqi.testproblem.Deconvolution2D() # Uncomment for 2D

# Some helper variables
n = TP.model.domain_dim # Number of parameters
N = TP.model.domain_geometry.shape[0] # Number of parameters in one dimension
ndim = len(TP.model.domain_geometry.shape) # Number of dimensions

# %% Define GMRF prior (structured Gaussian)
prior = cuqi.distribution.GMRF(
    mean=np.zeros(n),
    prec=500, # Higher precision means more regularized solution.
    order=1,  # Higher order mean more correlated structures. Can be 1 or 2.
    bc_type="zero", # Boundary conditions for GMRF. Non-zero are experimental.
    physical_dim=ndim,
    partition_size=N,
)

# Plot samples of prior
prior.sample(5).plot()

# %% Set prior and sample posterior
TP.prior = prior
samples = TP.sample_posterior(200)

# Plot samples of posterior
samples.plot_ci(exact=TP.exactSolution)

# %%
