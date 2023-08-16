"""
Random variables in CUQIpy
--------------------------

This tutorial introduces the concept of random variables and how to use
them in ``cuqipy``.

"""

from cuqi.testproblem import Deconvolution1D
from cuqi.distribution import Gaussian, Gamma, LMRF
from cuqi.problem import BayesianProblem
import numpy as np


######################################################################
# Introduction
# ~~~~~~~~~~~~
# 
# When defining a Distribution object in CUQIpy it can
# be viewed as a random variable.
# 
# In contrast to regular deterministic variables, random variables are not
# defined by a single value but by a probability distribution.
# 
# Consider the following example defining a random variable :math:`X` with
# a Gaussian (normal) distribution:
# 
# .. math::
# 
# 
#    X \sim \mathrm{Gaussian}(0,1)
# 
# This can be defined in CUQIpy as follows:
# 

X = Gaussian(0, 1)


######################################################################
# Random variables (and distributions) act different than regular Python
# variables in the following important ways:
# 
# 1. They are not immediately evaluated when defined. Instead, they are
#    evaluated in a lazy fashion in the context needed.
# 2. They define a probability distribution (single or multivariate)
#    instead of a single value or vector.
# 
# Despite these differences, random variables can be used in the same way
# as regular variables in most cases.
# 
# For example suppose we want to define a new variable :math:`Y` as
# follows:
# 
# .. math::
# 
# 
#    Y = (X + 10)^2
# 
# This can be done in CUQIpy as follows:
# 

Y = (X + 10)**2


######################################################################
# Calling print on ``Y`` reveals that it is a random variable which has
# recorded the operations performed on it and maintains a reference to the
# original random variable ``X``.
# 

print(Y)


######################################################################
# Evaluating random variables
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Currently, two modes of evaluation are supported for random variables:
# 
# 1. Sampling
# 2. Direct evaluation
# 
# Sampling
# ^^^^^^^^
# 
# Sampling works by drawing a random sample from the probability
# distribution of the random variable. This is done using the ``sample``
# method.
# 
# This is possible because the random variable knows of the
# transformations performed on it and can use this information to
# transform the sample drawn from the original distribution.
# 
# This means for example that we can both draw a sample from :math:`X` and
# :math:`Y` using the same syntax:
# 

print(X.sample())

print(Y.sample())


######################################################################
# Direct evaluation
# ^^^^^^^^^^^^^^^^^
# 
# Direct evaluation works by evaluating the random variable using the
# transformations performed on it. This is done using the ``__call__``
# method (which is called when using the ``()`` operator) in Python.
# 
# For example we can evaluate :math:`Y` at value ``X=3`` as follows:
# 

Y(3)


######################################################################
# **Note**. A third mode, probability density evaluation (e.g.asking
# for ``Y.logd``) is planned for future versions.
# 


######################################################################
# Algebraic operations on random variables
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Random variables support many basic algebraic operations such as
# addition, subtraction, multiplication, division, exponentiation, etc.
# 
# Random variables also support algebra between each other. For example we
# can define two new random variables :math:`x`, :math:`y` as follows:
# 

x = Gamma(1, 1)
y = Gaussian(0, 1)


######################################################################
# We can then define a new random variable :math:`z` as as the sum of
# :math:`x` and :math:`y^2`:
# 

z = x + (y**2+x**2)*0.1
print(z)


######################################################################
# Sampling and direct evaluation work as expected:
# 

print(z.sample())

z(x=1, y=2)


######################################################################
# Hierarchical modelling
# ~~~~~~~~~~~~~~~~~~~~~~
# 
# One of the main advantages of random variables is that they allow for
# hierarchical modelling of Bayesian problems.
# 
# Consider for example a 1D deconvolution problem as described in the
# CUQIpy tutorial Uncertainty Quantification in one-dimensional
# deconvolution
# 
# Here we have a forward model :math:`\mathbf{A}` defined by the
# expression :math:`\mathbf{y} = \mathbf{A}\mathbf{x}` where
# :math:`\mathbf{y}` is the observed data and :math:`\mathbf{x}` is the
# unknown signal.
# 
# We load a testproblem and observed data matching this description as
# follows:
# 

A, y_obs, info = Deconvolution1D.get_components(phantom="square", PSF_param=2)

print(A)

# Define n-dimensional zero vector
x_zero = np.zeros(A.domain_dim)


######################################################################
# Now we can define a hierarchical model for this Bayesian problem as
# follows:
# 
# .. math::
# 
# 
#    \begin{align*}
#    d &\sim \mathrm{Gamma}(1, 10^{-4} ) \\
#    s &\sim \mathrm{Gamma}(1, 10^{-4} ) \\
#    \mathbf{x} &\sim \mathrm{LMRF}(\mathbf{0}, d^{-1}) \\
#    \mathbf{y} &\sim \mathrm{Gaussian}(\mathbf{A}\mathbf{x}, s^{-1}) \\
#    \end{align*}
# 

d = Gamma(1, 1e-4)
s = Gamma(1, 1e-4)
x = LMRF(x_zero, 1/d)
y = Gaussian(A @ x, 1/s)


######################################################################
# For illustrative purposes let us put this into a Bayesian Problem
# object:
# 

BP = BayesianProblem(d, s, x, y)
print(BP)
