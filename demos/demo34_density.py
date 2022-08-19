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

# 2: The logd evaluates the log density, which only promises to be
# proportional to the logpdf. (In this case they are the same).
print(x.logd(1))

# %% Evaluating the logd of a conditioned distribution
# Suppose we have a conditional distribution
y = Normal(0, lambda s: s, name="y")

# We can still evaluate the log density simply by passing the value of the 
# conditioning variable as an argument also.
print(y.logd(s=1, y=1))

# In some cases we want to explicitly create a new object representing
# the conditioned distribution: p(y|s=1)
y_s = y(s=1) # This is equivalent to y._condition(s=1).
y_s

# %% Conditioning on the main parameter of a distribution
# In some cases we want to explicitly create a new object representing
# the conditioned distribution on the main parameter.
# That is, a likelihood L_{y=1}(s):=p(y=1|s)

# This is achieved using the same notation as above, but with the main parameter
# as the conditioning variable.
L_y = y(y=1) # This is equivalent to y._condition(y=1).
L_y

# %% Finally we can also completely specify a density by passing all
# unspecified variables as arguments, e.g. p(y=1|s=1)
y(y=1, s=1)

# %% Efficiency
# Suppose we wanted to evaluate p(y={1,2,3,4,5}|s=1). This can be achieved efficiently
# by first conditioning on y, and then evaluating the density at the different values of s.

y_s = y(s=1) # y(s=1) is only set once and computed
print(y_s(y=1))
print(y_s(y=2))
print(y_s(y=3))
print(y_s(y=4))
print(y_s(y=5))
# %%
