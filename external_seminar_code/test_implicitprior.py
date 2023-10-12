# %% Initialize and import CUQI
import sys
sys.path.append("..") 
import numpy as np
import matplotlib.pyplot as plt

import cuqi
from cuqi.distribution import Gaussian, ImplicitRegularizedGaussian
from cuqi.problem import BayesianProblem
from cuqi.sampler import RegularizedLinearRTO

#%% Load problem
n = 128
A, y_data, info = cuqi.testproblem.Deconvolution1D.get_components(dim=n, phantom='square')


#%% Minimal example
x  = ImplicitRegularizedGaussian(Gaussian(0.5*np.ones(n), 0.1), constraint = "nonnegative")
y  = Gaussian(A@x, 0.001)
BP = BayesianProblem(y, x).set_data(y=y_data)
BP.UQ(exact=info.exactSolution)

#%% More complicated example
x  = ImplicitRegularizedGaussian(Gaussian(0.5*np.ones(n), 0.1), constraint = "nonnegative")
y  = Gaussian(A@x, 0.001)

BP = BayesianProblem(y, x).set_data(y=y_data)
posterior = BP.posterior

sampler = RegularizedLinearRTO(posterior, maxit=100, stepsize = 5e-4, abstol=1e-10)
samples = sampler.sample(500, 100)

plt.figure()
samples.plot_median()
samples.plot_ci()
plt.plot(info.exactSolution)
plt.legend(["median", "CI", "mean"])
plt.show()


#%% Gibbs example
from cuqi.distribution import Gamma, JointDistribution
from cuqi.sampler import Gibbs, Conjugate

d = Gamma(1, 1e-4)
l = Gamma(1, 1e-4)
x = ImplicitRegularizedGaussian(Gaussian(0.5*np.ones(n), lambda d: d), constraint = "nonnegative")
y = Gaussian(A@x, lambda l: 1/l)

# Combine into a joint distribution
joint = JointDistribution(d, l, x, y)
posterior = joint(y=y_data)

cond = posterior(x=np.zeros(n), d=1)

sampling_strategy = {
    'x': RegularizedLinearRTO,
    'd': Conjugate,
    'l': Conjugate
}

sampler = Gibbs(posterior, sampling_strategy)
samples = sampler.sample(1000, 1000)

plt.figure()
samples['d'].plot_trace(figsize=(8,2))
plt.show()

plt.figure()
samples['l'].plot_trace(figsize=(8,2))
plt.show()

plt.figure()
samples['x'].plot_ci()
plt.show()