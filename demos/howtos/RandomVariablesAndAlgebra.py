"""
Random Variables and Algebra in CUQIpy
======================================

CUQIpy provides a simple algebraic framework for defining and manipulating
random variables.

In this example, we demonstrate how to define random variables, apply algebraic
operations on them, and finally use them in Bayesian Problems.
"""

#%%
# Defining Random Variables
# -------------------------
# Random variables can be defined by either initialising the RandomVariable class
# with a distribution object or by retrieving the `rv` attribute of a distribution.
# The distribution object can be any distribution from the `cuqi.distribution` module.

from cuqi.distribution import Normal
from cuqi.experimental.algebra import RandomVariable

x = RandomVariable(Normal(0, 1))
y = Normal(0, 1).rv

# %%
# Recording Algebraic Operations
# ------------------------------
# We can now perform some algebraic operations on the random variables. The
# operations are recorded in a computational graph, which can be evaluated
# later.

print("Basic operations: \n")
print(f"x + y yields:\n{x + y}\n")
print(f"x - y yields:\n{x - y}\n")
print(f"x * y yields:\n{x * y}\n")
print(f"x / y yields:\n{x / y}\n")

# %%
print("Complex operations: \n")
print(f"x**2 + 2*x*y + y**2 yields:\n{x**2 + 2*x*y + y**2}\n")
print(f"(x + y)**2 yields\n{(x + y)**2}\n")

# %%
print("Array operations: \n")
print(f"x[0] + y[1] yields:\n{x[0] + y[1]}\n")

# %%
# Utilizing the recorded operations
# ---------------------------------
# We can evaluate the recorded operations by calling the random variable object
# with the desired values for the random variables.

# Define a new random variable 'z'
z = (x + y)**2

# Evaluate the expression (using the __call__ method)
print(f"z={z.expression} evaluated at x=1, y=2 yields: {z(x=1, y=2)}")
# %%
# Building Bayesian Problems
# --------------------------
# Random variables can be used to define Bayesian problems. In this example we build
# an example Bayesian problem using the Deconvolution1D test problem.

from cuqi.testproblem import Deconvolution1D
from cuqi.distribution import Gaussian, Gamma, GMRF
from cuqi.experimental.algebra import RandomVariable
from cuqi.problem import BayesianProblem
from cuqi.distribution import JointDistribution
from cuqi.experimental.mcmc import HybridGibbs, LinearRTO, Conjugate, ConjugateApprox
import numpy as np
import matplotlib.pyplot as plt

# Forward model
A, y_obs, info = Deconvolution1D().get_components()

# Bayesian Problem (defined using Random Variables)
d = Gamma(1, 1e-4).rv
s = Gamma(1, 1e-4).rv
x = GMRF(np.zeros(A.domain_dim), d).rv
y = Gaussian(A @ x, 1/s).rv

# Combine into a Bayesian Problem and perform UQ
BP = BayesianProblem(y, x, s, d)
BP.set_data(y=y_obs)
BP.UQ(exact={"x": info.exactSolution})

# Random variables can also be used to define JointDistribution. Here we solve the same
# problem above by explictly forming a target distribution and then drawing samples with
# the HybridGibbs sampler.
target = JointDistribution(y, x, s, d)(y=y_obs)

# Sampling strategy
sampling_strategy = {
    "x" : LinearRTO(),
    "s" : Conjugate(),
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

# %%
# Conditioning on random variables (example 1)
s = Gaussian(0, 1).rv
x = Gaussian(0, s).rv
y = Gaussian(0, lambda d: d).rv

z = x+y

z.condition(s=1)
z.condition(d=2)

# %%
# Or conditioning on the variables s, or d
z.condition(s=1)

# %%
# Conditioning on random variables (example 2)
from cuqi.testproblem import Deconvolution1D
from cuqi.distribution import Gaussian, Gamma, GMRF
from cuqi.experimental.algebra import RandomVariable
from cuqi.problem import BayesianProblem
import numpy as np

# Forward model
A, y_obs, info = Deconvolution1D(dim=4).get_components()

# Bayesian Problem (defined using Random Variables)
d = Gamma(1, 1e-4).rv
s = Gamma(1, 1e-4).rv
x = GMRF(np.zeros(A.domain_dim), d).rv
y = Gaussian(A @ x, 1/s).rv


z = x+y

z.condition(x=np.zeros(A.domain_dim))

# %%
