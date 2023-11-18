"""
Random variables in CUQIpy
--------------------------

This tutorial introduces the concept of random variables and how to use
them in ``CUQIpy``. At this point in time random variables are still
experimental and the API is subject to change.

"""
#%%
from cuqi.testproblem import Deconvolution1D
from cuqi.distribution import Gaussian, Gamma, LMRF
from cuqi.problem import BayesianProblem


# %%
######################################################################
# Introduction
# ~~~~~~~~~~~~
# 
# When defining a Distribution object in CUQIpy it can
# be converted to a random variable after it has been defined.
# 
# In contrast to regular deterministic variables, random variables are not
# defined by a single value, but by a probability distribution.
# 
# Consider the following example defining a random variable :math:`x` from
# a Gaussian (normal) distribution X:
# 
# .. math::
# 
#    x \sim \mathrm{Gaussian}(0,1)
# 
# In CUQIpy the random variable is defined by first creating a distribution
# and subsequently generating a random variable from it using the ``rv``
# attribute as follows:
# 

X = Gaussian(0, 1) # Distribution

x = X.rv # Random variable from distribution

print(x)

# %%
######################################################################
# This can also be done in a single line as follows:
#

x = Gaussian(0, 1).rv

# %%
######################################################################
# The difference between distributions and random variables can be subtle.
# From a mathematical point of view, a random variable is a function from
# a probability space to a measurable space, while a distribution is a
# measure on a measurable space.
#
# In the context of CUQIpy, a distribution is a Python object which
# represents a probability distribution. Probability distributions are defined
# by their probability density function (PDF). They support sampling from
# the distribution and evaluating the PDF at a given point.
#
# In the context of CUQIpy, a random variable is a Python object which
# represents a random variable for the purpose of defining a Bayesian
# problem. Random variables are defined by their underlying distribution
# and the algebraic operations performed on them.
#
# For simple random variable, the underlying distribution can be accessed using
# the ``dist`` attribute.

print(x.dist)


# %%
######################################################################
# Random variables act different than regular Python
# variables in the following important ways:
# 
# 1. They are not immediately evaluated when defined. Instead, they are
#    evaluated in a lazy fashion in the context needed.
# 2. They contain at least one probability distribution (single or multivariate)
#    instead of a single value or vector of values.
# 
# Despite these differences, random variables can be used in the same way
# as regular variables in most cases.
# 
# For example suppose we want to define a new variable :math:`y` as
# follows:
# 
# .. math::
# 
# 
#    y = (x + 10)^2
# 
# This can be done in CUQIpy as follows:
# 

y = (x + 10)**2


# %%
######################################################################
# Calling print on ``y`` reveals that it is a random variable which has
# recorded the operations performed on it and maintains a reference to the
# random variable ``x`` and its underlying distribution.
# 

print(y)

# %%
######################################################################
# Transformed random variables
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The random variable ``y`` defined above is an example of a transformed
# random variable. Transformed random variables are random variables which
# are defined by applying algebraic operations to other random variables.
# 
# In CUQIpy, it is possible to determine if a random variable is
# transformed by checking the ``is_transformed`` attribute.
#

print(y.is_transformed)


# %%
######################################################################
# Evaluating random variables
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Currently, a two modes of evaluation are supported for all
# (simple and transformed) random variables:
# 
# 1. Sampling
# 2. Direct evaluation
# 
# Sampling
# ^^^^^^^^
# 
# Sampling works by drawing a random sample from the underlying probability
# distribution of the random variable. This is done using the ``sample``
# method.
# 
# This is possible because the random variable knows of the
# transformations performed on it and can use this information to
# transform the sample drawn from the original distribution.
# 
# This means for example that we can both draw a sample from :math:`x` and
# :math:`y` using the same syntax, which also matches the syntax used for
# sampling from distributions.
# 

# Draws a sample from x ~ Gaussian(0,1)
print(x.sample()) 

# Draws a sample from y = (x+10)^2, x ~ Gaussian(0,1)
print(y.sample()) 


# %%
######################################################################
# Direct evaluation
# ^^^^^^^^^^^^^^^^^
# 
# Direct evaluation is defined by the transformations performed on it.
# It works by evaluating the transformations given a an input value.
# The ``__call__`` method (which is called when using the ``()`` operator)
# in Python is used for this purpose.
# 
# For example we can evaluate :math:`y` at value ``x=3`` as follows:
# 

# Evaluates y = (x+10)^2 at x=3
y(3)


# %%
######################################################################
# Other methods and attributes of random variables
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In addition to the abovementioned evaluation methods, simple, non-transformed
# random variables support a variety of other methods and attributes for accessing
# the underlying distribution. These include:
#
# 1. ``logd``: Evaluates the log of the probability density function of
#    the underlying distribution for the random variable at a given point.
# 2. ``dim``: Returns the dimension of the random variable.
# 3. ``geometry``: Returns the geometry of the random variable.
# 4. ``gradient``: Returns the gradient og the log density function of
#    the random variable at a given point.
# 5. ``condition``: Returns a new random variable where the underlying
#    distribution has been conditioned on the given input.
# 
# These methods and attributes are demonstrated at the end of this tutorial.
#


# %%
######################################################################
# Algebraic operations on random variables
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Random variables support many basic algebraic operations such as
# addition, subtraction, multiplication, division, exponentiation, etc.
# 
# Random variables also support algebra between each other. For example we
# can define two new variables :math:`x`, :math:`y` as follows:
#

x = Gamma(1, 1).rv
y = Gaussian(0, 1).rv


# %%
######################################################################
# We can then define a new random variable :math:`z` as follows:
# 

z = x + (y**2+x**2)*0.1
print(z)

# %%
######################################################################
# Sampling and direct evaluation work as expected:
# 

# Draws a sample from z with x and y sampled from their respective distributions
print(z.sample()) 

# Evaluates z at x=1, y=2
z(x=1, y=2) 


# %%
######################################################################
# Hierarchical modelling
# ~~~~~~~~~~~~~~~~~~~~~~
# 
# One of the main advantages of random variables is that they enable
# a succinct syntax for hierarchical modelling of Bayesian problems.
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

A, y_obs, info = Deconvolution1D(phantom="square").get_components()

print(A)

# %%
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
# Noting that the algebraic operations performed on the random variables
# are used to define the conditional distributions used in the subsequent
# random variables (e.g. the covariance of y is lazily evaluated as 1/s).

d = Gamma(1, 1e-4).rv
s = Gamma(1, 1e-4).rv
x = LMRF(0, 1/d, geometry=A.domain_geometry).rv
y = Gaussian(A @ x, 1/s).rv

# %%
######################################################################
# For illustrative purposes let us put this into a Bayesian Problem
# object:
# 

BP = BayesianProblem(d, s, x, y)
print(BP)

# %%
######################################################################
# Providing data to the Bayesian problem creates a posterior as shown.
#

BP.set_data(y=y_obs)
print(BP)

# %%
######################################################################
# Given the Bayesian problem posterior we can now sample from
# the posterior distribution using MCMC as follows:
#

samples = BP.sample_posterior(1000)

# %%
######################################################################
# We can now plot a 95 credibility interval of the samples for x
# from the posterior distribution as follows:
#

samples["x"].plot_ci(exact=info.exactSolution)

# %%
# Revisiting the random variable methods and attributes
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In this section we revisit the methods and attributes of random variables
# introduced earlier in this tutorial.
# 
# For this purpose let us define a new random 3-dimensional
# variable :math:`x` as follows:

x = Gaussian(0, 1, geometry=3).rv

# %%
######################################################################
# Probability density evaluation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Probability density evaluation works by evaluating the probability
# density function of the random variable at a given point. This is done
# using the ``logd`` method. **Note**. Only simple (non-transformed)
# random variables support this method at this point in time.
#

x.logd([1, 2, 3])

# %%
######################################################################
# Gradient evaluation
# ^^^^^^^^^^^^^^^^^^^
#
# Gradient evaluation works by evaluating the gradient of the log
# probability density function of the random variable at a given point.
#

x.gradient([1, 2, 3])


# %%
######################################################################
# Dimension and geometry
# ^^^^^^^^^^^^^^^^^^^^^^
#
# The dimension of a random variable can be accessed using the ``dim``
# attribute. The geometry of a random variable can be accessed using the
# ``geometry`` attribute.
#

print(x.dim)
print(x.geometry)

# %%
######################################################################
# Conditioning
# ^^^^^^^^^^^^
#
# Conditioning works by conditioning the underlying distribution of the
# random variable on the given input. This is done using the ``condition``
# method. For more details see documentation of conditioning on distributions.

z = Gaussian(0, lambda s: s).rv

print(z)
print(z.condition(s=10))

