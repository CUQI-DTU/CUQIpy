"""
CUQIpy Basics
=============

In this tutorial, we will learn about the basic concepts of CUQIpy.

The aim of this tutorial is to help you get started with CUQIpy and start doing uncertainty quantification in five simple steps.

    * Step 1: Define deterministic model
    * Step 2: Define prior(s)
    * Step 3: Define likelihood
    * Step 4: Sample from the posterior
    * Step 5: Analyze the results

"""

# %% First we import the modules needed.
import cuqi

# %%
# Step 1: Deterministic model
# ---------------------------
#
# For the purpose of this tutorial we focus on a simple 1D deconvolution problem.
#
# Consider the deterministic inverse problem
#
# .. math::
#
#   \mathbf{y} = \mathbf{A} \mathbf{x}
#
# where :math:`\mathbf{A}` is a matrix representing a 1D convolution operation and
# :math:`\mathbf{y}` and :math:`\mathbf{x}` are the data and unknown (solution to the inverse problem) respectively.
#
# A linear forward model like :math:`\mathbf{A}` is represented by a :class:`~cuqi.model.LinearModel`
# and any data (like some observed data :math:`\mathbf{y}^\mathrm{obs}`) as a :class:`~cuqi.array.CUQIarray`.
#
# The easiest way to get these two components is to use the built-in testproblem module.
# If you are interested in defining your own model see the how-to guides on that.
# 
# Let us extract the model and data for a 1D deconvolution.
# In this case we use the default settings for the testproblem.

A, y_obs, info = cuqi.testproblem.Deconvolution1D().get_components()

# %%
# .. todo::
#    Continue tutorial here.
