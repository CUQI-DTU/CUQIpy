"""
Random variables
================

    This tutorial shows how to define and use random variables in CUQIpy.

"""
# %%
# Setup
# -----
# First, we import the necessary packages.

import cuqi
from cuqi.distribution import Gaussian, Gamma

# %%
# Random variables are born from distributions.
# In the most basic case, we can simply apply algebraic operations
# to distributions to create random variables.

x = cuqi.distribution.Gaussian(0, 1) # x ~ N(0, 1)

print(x+1) # returns a RandomVariable object with operation recorded

# %%
# We can also create random variables directly from distributions.

x_rv = x.as_random_variable() # returns a RandomVariable object

print(x_rv)

# %%
# Algebraic operations can be combined to create more complex random variables.

print( -(abs((x + 10)**2 - 100 )) )

# %%
# Random variables record operations and can be played back later via the __call__ method.

x_transformed = (x + 1)**2

x_transformed(2) # returns (2+1)^2 = 9

# %%
# The basic usage of random variables is to more easily define Bayesian problems.
# Consider this 1D deconvolution problem example

# Imports
from cuqi.testproblem import Deconvolution1D
from cuqi.distribution import Gaussian, Gamma, GMRF
from cuqi.problem import BayesianProblem
import numpy as np

# Forward model
A, y_obs, info = Deconvolution1D.get_components()

# Bayesian problem
d = Gamma(1, 1e-4)
s = Gamma(1, 1e-4)
x = GMRF(np.zeros(A.domain_dim), d)
y = Gaussian(A @ x, 1/s)

BP = BayesianProblem(y, x, s, d)
print(BP)

# %%
# Notice in the example above that the distribution was directly used to define another distribution
# for both x and y using d and s respectively as dependencies.

