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
# Random variables can be defined using the RandomVariable class. The RandomVariable
# class requires a distribution object to be passed as an argument. The distribution
# object can be any distribution object from the `cuqi.distribution` module.

from cuqi.distribution import Normal
from cuqi.experimental.algebra import RandomVariable


x = RandomVariable(Normal(0, 1))
y = RandomVariable(Normal(0, 1))

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
import numpy as np

# Forward model
A, y_obs, info = Deconvolution1D().get_components()

# Bayesian Problem (defined using Random Variables)
d = RandomVariable(Gamma(1, 1e-4))
s = RandomVariable(Gamma(1, 1e-4))
x = RandomVariable(GMRF(np.zeros(A.domain_dim), d))
y = RandomVariable(Gaussian(A @ x, 1/s))

# Combine into a Bayesian Problem and perform UQ
BP = BayesianProblem(y, x, s, d)
BP.set_data(y=y_obs)
BP.UQ(exact={"x": info.exactSolution})

# %%
# Conditioning on random variables (example 1)
from cuqi.distribution import Gaussian
from cuqi.experimental.algebra import RandomVariable

x = RandomVariable(Gaussian(0, lambda s: s))
y = RandomVariable(Gaussian(0, lambda d: d))

z = x+y

z.condition(x=1)

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
d = RandomVariable(Gamma(1, 1e-4))
s = RandomVariable(Gamma(1, 1e-4))
x = RandomVariable(GMRF(np.zeros(A.domain_dim), d))
y = RandomVariable(Gaussian(A @ x, 1/s))


z = x+y

z.condition(x=np.zeros(A.domain_dim))

# %%
# Sampling from random variables
# ------------------------------
# Random variables can be sampled using the `sample` method. The method returns a
# sample from the distribution of the random variable.

x = RandomVariable(Normal(0, 1))

print(f"Sample from x: {x.sample()}")

# %%
# This can be combined with algebraic operations to sample from more complex
# random variables.

z = x + x**2 + 25

print(f"Sample from z: {z.sample()}")

# %%
# Constructing a Beta distribution using Gamma random variables
# -------------------------------------------------------------
# Random variables can also be combined to create new distributions.
# This is primarily useful for sampling at this stage.
# For example, a Beta distribution can be constructed from two Gamma distributions:
# If X ~ Gamma(a1, 1) and Y ~ Gamma(a2, 1), then Z = X / (X + Y) ~ Beta(a1, a2).
# We illustrate this by comparing samples from a Beta distribution to samples
# constructed using two Gamma distributions.

from cuqi.distribution import Beta, Gamma

# Define the shape parameters of the Beta distribution
a1, a2 = 3, 2

# Step 1: Directly define a Beta distribution
z_ref = RandomVariable(Beta(a1, a2))

# Step 2: Construct the Beta distribution using Gamma random variables
x = RandomVariable(Gamma(a1, 1))  # X ~ Gamma(a1, 1)
y = RandomVariable(Gamma(a2, 1))  # Y ~ Gamma(a2, 1)
z = x / (x + y)                   # Z ~ Beta(a1, a2)

# Step 3: Sample from both distributions
z_samples = z.sample(10000)       # Samples from constructed Beta distribution
z_ref_samples = z_ref.sample(10000)  # Samples from direct Beta distribution

# Step 4: Plot histograms of the samples for comparison
z_samples.hist_chain([0], bins=100)
z_ref_samples.hist_chain([0], bins=100)
