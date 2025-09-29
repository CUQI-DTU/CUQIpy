# %%
# This demo shows the use of RandomVariable (RV) as prior in posteior sampling.
# Specifically, we test the use of RV as prior through 
# 1) BayesianProblem,
# 2) JointDistribution with MALA,
# 3) JointDistribution with Gibbs (NUTS + Conjugate) and
# 4) JointDistribution with Regularized LinearRTO.

from cuqi.testproblem import Deconvolution1D
from cuqi.distribution import Gaussian, Gamma, GMRF
from cuqi.problem import BayesianProblem
from cuqi.distribution import JointDistribution
from cuqi.implicitprior import NonnegativeGMRF
from cuqi.sampler import HybridGibbs, Conjugate, MALA, NUTS, RegularizedLinearRTO
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1912)

# %% Forward model
n = 128
A, y_obs, info = Deconvolution1D(dim=n, phantom='square').get_components()

# %% 1) Bayesian problem
x = GMRF(np.zeros(A.domain_dim), 100).rv
y = Gaussian(A @ x, 1/10000).rv

# Combine into a Bayesian Problem and perform UQ
BP = BayesianProblem(y, x)
BP.set_data(y=y_obs)
BP.UQ(exact=info.exactSolution)

# %% 2) Joint distribution with MALA
target = JointDistribution(y, x)(y=y_obs)
sampler = MALA(target, scale=0.0002, initial_point=np.zeros(n))
sampler.sample(10000)
samples = sampler.get_samples()
plt.figure()
samples.plot_ci(exact=info.exactSolution)

# %% 3) Hybrid Gibbs (NUTS + Conjugate)
d = Gamma(1, 1e-4).rv
x = GMRF(np.zeros(A.domain_dim), d).rv
y = Gaussian(A @ x, 1/10000).rv

target = JointDistribution(y, x, d)(y=y_obs)

# Sampling strategy
sampling_strategy = {
    "x" : NUTS(),
    "d" : Conjugate()
}

# Gibbs sampler
sampler = HybridGibbs(target, sampling_strategy)
# Run sampler
sampler.warmup(200)
sampler.sample(1000)
samples = sampler.get_samples()

# Plot
plt.figure()
samples["x"].plot_ci(exact=info.exactSolution)

# %% 4) Regularized GMRF
x = NonnegativeGMRF(np.zeros(A.domain_dim), 50).rv
y = Gaussian(A @ x, 1/10000).rv

target = JointDistribution(y, x)(y=y_obs)

# Regularized Linear RTO sampler
sampler = RegularizedLinearRTO(target)
# Run sampler
sampler.sample(1000)
samples = sampler.get_samples()

# Plot
plt.figure()
samples.plot_ci(exact=info.exactSolution)
