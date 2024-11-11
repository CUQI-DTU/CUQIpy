# %%
import numpy as np
import matplotlib.pyplot as plt
import sys

from cuqi.model import AffineModel, LinearModel
from cuqi.experimental.mcmc import MH, LinearRTO
from cuqi.distribution import Gaussian, LMRF, GMRF, JointDistribution
from cuqi.geometry import Discrete, Continuous1D
from cuqi.array import CUQIarray

# %% Test problem
### y = Ax + b
### Infer x, given the model A and b, and noisy observations of y
### There are two ways to solve this problem: 
### 1) either you use the linear model only y=Ax realizing that you can subtract the bias from the data or
### 2) define an affine model and let CUQIpy handle it.
### This script demonstrates that both methods provide the same results.

# Dimensions
n = 2
m = 5

# Unknown parameter
x_truevec = np.array([3, 2])
x_true = CUQIarray(x_truevec, geometry=Discrete(["x1", "x2"]))

# Observation times
t = np.linspace(0, m, m, endpoint=False)

# Forward model
Amat = np.ones((m, n))
Amat[:, 1] = t

# shift/bias
b = np.ones(m) * 3

# Geometries
domain_geometry = Discrete(n)
range_geometry = Continuous1D(t)

# Prior
x_mean = np.array([2.0, 1.0])
x_std = np.ones(n)
x = Gaussian(x_mean, sqrtcov=x_std, geometry=domain_geometry)

# Noisy data
np.random.seed(10)
y_std = np.ones(m)
y_obs = Gaussian(Amat @ x_truevec + b, sqrtcov=y_std, geometry=range_geometry).sample()

# %% Linear model, include bias (b) from model in data term such that data = y-b

# CUQI Model
A = LinearModel(Amat, domain_geometry=domain_geometry, range_geometry=range_geometry)

# Likelihood
y = Gaussian(A @ x, sqrtcov=y_std, geometry=range_geometry)

# Posterior
posterior = JointDistribution(x, y)(y=y_obs - b)

# Sample posterior with linear RTO
np.random.seed(1000000)
Linearsampler = MH(posterior, initial_point=x_mean)  # Init point must be float (NOT AFFINE MODEL SPECIFIC)
Linearsampler.warmup(200)
Linearsampler.sample(10000)
samplesLinear = Linearsampler.get_samples()
samplesLinear_burnin = samplesLinear.burnthin(200)

# Compute posterior mean and variance and plot
meanLinear = samplesLinear_burnin.mean()
varLinear = samplesLinear_burnin.variance()

print("Posterior mean and variance using Linear model: ")
print("mean: {}".format(meanLinear))
print("cov: {}".format(varLinear))

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))
ax.plot(t, A(samplesLinear_burnin).mean(), linestyle="--", label="Sample mean using Linear model")
ax.plot(t, y_obs - b, "o", label="Noisy data")
ax.set_xlabel("t")
ax.set_ylabel("y")
plt.legend()

cs = samplesLinear_burnin.plot_pair(marginals=True)
plt.suptitle("Posterior samples using Linear model")


# %% Affine model, include bias (b) in affine model such that data = y

# CUQI Model
affine_model = AffineModel(Amat, b, domain_geometry=domain_geometry, range_geometry=range_geometry)

# Likelihood
y = Gaussian(affine_model(x), sqrtcov=y_std, geometry=range_geometry)

# Posterior
posterior = JointDistribution(x, y)(y=y_obs)

print(posterior.likelihood.model.forward(np.array([1, 1])))
print(posterior.likelihood.model._forward_func_no_shift(np.array([1, 1])))

# Sample posterior with MH
np.random.seed(1000000)
Affinesampler = MH(posterior, initial_point=x_mean)  # Init point must be float (NOT AFFINE MODEL SPECIFIC)
Affinesampler.warmup(200)
Affinesampler.sample(10000)
samplesAffine = Affinesampler.get_samples()
samplesAffine_burnin = samplesAffine.burnthin(200)

# Compute posterior mean and variance and plot
meanAffine = samplesAffine_burnin.mean()
varAffine = samplesAffine_burnin.variance()

print("Posterior mean and variance from Affine model: ")
print("mean: {}".format(meanAffine))
print("cov: {}".format(varAffine))

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))
ax.plot(t, affine_model(samplesAffine_burnin).mean(), linestyle="--", label="Sample mean using Affine model")
ax.plot(t, y_obs, "o", label="Noisy data")
ax.set_xlabel("t")
ax.set_ylabel("y")
plt.legend()


cs = samplesAffine_burnin.plot_pair(marginals=True)
plt.suptitle("Posterior samples using Affine model")

# %%
# Compare the results from the two models
# They should provide exactly the same results as the seed is the same
assert np.allclose(samplesLinear.samples, samplesAffine.samples)
