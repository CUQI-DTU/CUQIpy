"""
Uncertainty Quantification in one-dimensional deconvolution
===========================================================

    This tutorial walks through the process of solving a simple 1D 
    deconvolution problem in a Bayesian setting. It also shows how
    to define such a convolution model in CUQIpy.

"""
# %%
# Setup
# -----
# We start by importing the necessary modules

import cuqi
import numpy as np
import matplotlib.pyplot as plt

# %%
# Setting up the forward model
# ----------------------------
# We start by defining the forward model. In this case, we will use a simple
# convolution model. The forward model is defined by the following equation:
#
# .. math::
#    \mathbf{y} = \mathbf{A} \mathbf{x}
#
# where :math:`\mathbf{y}` is the data, :math:`\mathbf{A}` is the convolution (forward model)
# operator, and :math:`\mathbf{x}` is the solution.
#
# The easiest way to define the forward model is to use the testproblem module.
# This module contains a number of pre-defined test problems that contain the
# forward model and synthetic data. In this case, we will use the
# :class:`cuqi.testproblem.Deconvolution1D` test problem. We extract the forward model
# and synthetic data from the test problem by calling the :func:`get_components`
# method.

# Forward model and data
A, y_data, info = cuqi.testproblem.Deconvolution1D().get_components()

# %%
# There are many parameters that can be set when creating the test problem. For more details
# see the :class:`cuqi.testproblem.Deconvolution1D` documentation. In this case, we will use
# the default parameters. The :func:`get_components` method returns the forward model, 
# synthetic data, and a :class:`~ProblemInfo` object that contains information about the 
# test problem.
#
# Let's take a look at the forward model

print(A)

# %%
# We see that the forward model is a a :class:`~cuqi.model.LinearModel` object. This
# object contains the forward model and the adjoint model. We also see that the domain and
# range of the forward model are both continuous 1D spaces. Finally, we see that the default
# forward parameters are set to :math:`\mathbf{x}`.

# %%
# Let's take a look at the synthetic data and compare with the exact solution
# that we can find in the :class:`~ProblemInfo` object.

y_data.plot(label="Synthetic data")
info.exactSolution.plot(label="Exact solution")
plt.title("Deconvolution 1D problem")
plt.legend()

# %%
# Setting up the prior
# --------------------
#
# We now need to define the prior distribution for the solution. In this case, we will use
# a Gaussian Markov Random Field (GMRF) prior. For more details on the GMRF prior, see the
# :class:`cuqi.distribution.GMRF` documentation.

x = cuqi.distribution.GMRF(np.zeros(A.domain_dim), 200)


# %%
# Setting up the likelihood
# -------------------------
#
# We now need to define the likelihood. First let us take a look at the information provided
# by the test problem.

print(info.infoString)

# %%
# We see that the noise level is known and that the noise is Gaussian. We can use this
# information to define the likelihood. In this case, we will use a :class:`~cuqi.distribution.Gaussian`
# distribution.

y = cuqi.distribution.Gaussian(A @ x, 0.01**2)

# %%
# Bayesian problem (Joint distribution)
# -------------------------------------
#
# After defining the prior and likelihood, we can now define the Bayesian problem. The
# Bayesian problem is defined by the joint distribution of the solution and the data.
# This can be seen when we print the Bayesian problem.

BP = cuqi.problem.BayesianProblem(y, x)

print(BP)

# %%
# Setting the data (posterior)
# ----------------------------
#
# Now to set the data, we need to call the :func:`~cuqi.problem.BayesianProblem.set_data`

BP.set_data(y=y_data)

print(BP)

# %%
# Sampling from the posterior
# ---------------------------
#
# We can then use the automatic sampling method to sample from the posterior distribution.

samples = BP.sample_posterior(1000)

# %%
# Plotting the results
# --------------------

samples.plot_ci(exact=info.exactSolution)

# %%
# Unknown noise level
# -------------------
# In the previous example, we assumed that we knew the noise level of the data. In
# many cases, this is not the case. If we do not know the noise level, we can
# use a :class:`~cuqi.distribution.Gamma` distribution to model the noise level.

s = cuqi.distribution.Gamma(1, 1e-4)

# %%
# Update likelihood with unknown noise level
# ------------------------------------------

y = cuqi.distribution.Gaussian(A @ x, prec=lambda s: s)

# %%
# Bayesian problem (Joint distribution)
# -------------------------------------

BP = cuqi.problem.BayesianProblem(y, x, s)

print(BP)

# %%
# Setting the data (posterior)
# ----------------------------
#

BP.set_data(y=y_data)

print(BP)

# %%
# Sampling from the posterior
# ---------------------------

samples = BP.sample_posterior(1000)


# %%
# Plotting the results
# --------------------
#
# Let is first look at the estimated noise level
# and compare it with the true noise level

samples["s"].plot_trace(exact=1/0.01**2)

# %%
# We see that the estimated noise level is close to the true noise level. Let's
# now look at the estimated solution


samples["x"].plot_ci(exact=info.exactSolution)


# %%
# We can even plot traces of "x" for a few cases and compare
samples["x"].plot_trace(exact=info.exactSolution)

# %%
# And finally we note that the UQ method does this analysis automatically and shows a selected number of plots
BP.UQ(exact={"x": info.exactSolution, "s": 1/0.01**2})

