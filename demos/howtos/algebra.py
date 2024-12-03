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
print(f"z={z.formula} evaluated at x=1, y=2 yields: {z(x=1, y=2)}")
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
