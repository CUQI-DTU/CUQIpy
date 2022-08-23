# %% Overview of densities

# This demo showcases how the Density class adds a unified interface to conditioning and log density evaluation
# for Distributions and Likelihoods. The Density class is an abstract base class that is inherited by the
# Distribution and Likelihood classes. The Density class is not meant to be used directly.
import sys; sys.path.append("..")
from cuqi.distribution import Normal

# Most of these features are useful for the Gibbs sampler implementation down the line.
# The user may not be exposed or using most of these features directly.

# %%
# First we create a simple distribution
# We explicitly add the name to name the RV associated with the distribution
x = Normal(0, 1, name="x") # x ~ N(0, 1). Later we dont need to specify the name

# %% logpdf and logd

# Distributions expose two methods for evaluating its density function.
# 1: The logpdf evaluates the log probability density function at a point
print(x.logpdf(1))

# 2: The logd evaluates the un-normalized log density function, which only promises
# to be equal to the logpdf up to an additive constant. (In this case they are the same).
print(x.logd(1))

# %% Evaluating the logd of a conditioned distribution
# Suppose we have a conditional distribution p(y|s) = N(y|0,s^2)
y = Normal(0, lambda s: s, name="y")

# In some cases we want to explicitly create a new object representing
# the conditioned distribution: p(y|s=1) = N(y|0,1^2).
# For succinctness this is achieved using the short-hand __call__ method
# which conditions a density on a given set of parameters.
y_s = y(s=1) # This is equivalent to y._condition(s=1).

# We can then evaluate the logd of the conditioned distribution.
print(y_s.logd(1))

# %% Another short-hand for evaluating the logd of a conditioned distribution
# We can also evaluate the log density simply by passing the value of the 
# conditioning variable as an argument also.
# In this case we do not get access to y_s directly if we wanted to use it later.
print(y.logd(s=1, y=1))

# Note we use the keyword "y" to specify the name of the main parameter.

# %% Conditioning on the main parameter of a distribution
# In some cases we want to explicitly create a new object representing
# the conditioned distribution on the main parameter instead of simply
# evaluating the logd.

# That is e.g. the case for a likelihood L_{y=1}(s):=p(y=1|s)
# This is achieved using the same notation as above, but with the main parameter
# as the conditioning variable.
L_y = y(y=1) # This is equivalent to y._condition(y=1).
L_y

# %% Finally we can also specify *all* parameters of a density when conditioning.
# In this case the result is simply a constant, e.g. p(y=1|s=1) = constant.

# When conditioning on all parameters, an "EvaluatedDensity" is returned,
# representing that constant. The evaluated density is useful because
# it exposes the same interface as a density (logd, dim etc.) simplifying
# code for samplers and other modules.
y(y=1, s=1)

# %% Extracting the value of the EvaluatedDensity
# The value of the EvaluatedDensity can be extracted using the .value property.
y(y=1, s=1).value

# %% Efficiency considerations
# Suppose we wanted to evaluate p(y=i}|s=1) for i=1,...,5. This can be achieved efficiently
# by first conditioning on s=1, and then evaluating the density at the different values of s.

y_s = y(s=1) # y(s=1) is only set once and computed
for i in range(1, 6):
    print(y_s.logd(y=i))
# %%
