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

# %%
# Random variables are born from distributions.

x = cuqi.distribution.Gaussian(0, 1)

# %%
# Create random variable
y = cuqi.randomvariable.RandomVariable(x)

# %%
# Now we can define a new random variable from this distribution.

z = 1/y


# %%
