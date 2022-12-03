"""
How to define a Posterior distribution
======================================

The recommended way to define a posterior distribution is to use the
:class:`cuqi.distribution.JointDistribution` class to define the joint
distribution of the parameters and the data and then condition on observed
data to obtain the posterior distribution as shown in the examples below.
"""

# %%
# Imports
import cuqi
import numpy as np

# %%
# A simple Bayesian inverse problem
# --------------------------------
#
# .. math::
#    \mathbf{y} = \mathbf{A}\mathbf{x}
#
# See :class:~`cuqi.testproblem.Deconvolution1D` for more details.
#
# .. math::
#    \mathbf{x} \sim \mathcal{N}(\mathbf{0}, 0.2\mathbf{I})\\
#    \mathbf{y} \sim \mathcal{N}(\mathbf{A}\mathbf{x}, 0.1\mathbf{I})
#

# Forward model and data
A, y_obs, _ = cuqi.testproblem.Deconvolution1D.get_components()

# Bayesian model
x = cuqi.distribution.Gaussian(np.zeros(A.domain_dim), 0.1)  # x ~ N(0,1)
y = cuqi.distribution.Gaussian(A(x), 0.05**2)                # y ~ N(A(x), 0.05^2)

# %% Joint distribution
joint = cuqi.distribution.JointDistribution(x, y)              # p(x,y)
print(joint)

# %% Posterior distribution
posterior = joint(y=y_obs)                                     # p(x|y=y_obs)
print(posterior)


# %%
# One parameter, two forward models
# ---------------------------------
#
# .. math::
#    \mathbf{y} = \mathbf{A}\mathbf{x}\\
#    \mathbf{d} = \mathbf{B}\mathbf{x}\\
#
# .. math::
#    \mathbf{x} \sim \mathcal{N}(\mathbf{0}, 0.2\mathbf{I})\\
#    \mathbf{y} \sim \mathcal{N}(\mathbf{A}\mathbf{x}, 0.1\mathbf{I})\\
#    \mathbf{d} \sim \mathcal{N}(\mathbf{B}\mathbf{x}, 0.3\mathbf{I})

# Two forward models, same x generated the data
A, y_obs, _ = cuqi.testproblem.Deconvolution1D.get_components()
B, d_obs, _ = cuqi.testproblem.Deconvolution1D.get_components(kernel="sinc", noise_std=0.01)

# Define distributions
x = cuqi.distribution.Gaussian(np.zeros(A.domain_dim), 0.1)  # x ~ N(0,1)
y = cuqi.distribution.Gaussian(A(x), 0.05**2)                # y ~ N(A(x), 0.05^2)
d = cuqi.distribution.Gaussian(B(x), 0.01**2)                # d ~ N(B(x), 0.5^2)

# %% Joint distribution
joint2 = cuqi.distribution.JointDistribution(x, y, d)          # p(x,y,d)
print(joint2)

# %% Posterior distribution
posterior2 = joint2(y=y_obs, d=d_obs)                           # p(x|y=y_obs, d=d_obs)
print(posterior2)

# %%
# Multiple parameters, multiple forward models
# --------------------------------------------
#
# .. math::
#    \mathbf{y} = \mathbf{A}\mathbf{x}\\
#    \mathbf{d} = \mathbf{B}\mathbf{x}\\
#
# .. math::
#    \mathbf{l} \sim \mathrm{Gamma}(1, 1)\\
#    \mathbf{s} \sim \mathrm{Gamma}(0, 10^{-2})\\
#    \mathbf{x} \sim \mathcal{N}(\mathbf{0}, l^{-1}\mathbf{I})\\
#    \mathbf{y} \sim \mathcal{N}(\mathbf{A}\mathbf{x}, s^{-1}\mathbf{I})\\
#    \mathbf{d} \sim \mathcal{N}(\mathbf{B}\mathbf{x}, 0.3\mathbf{I})

# Two forward models, same x generated the data
A, y_obs, _ = cuqi.testproblem.Deconvolution1D.get_components()
B, d_obs, _ = cuqi.testproblem.Deconvolution1D.get_components(kernel="sinc", noise_std=0.01)

# Define distributions
l = cuqi.distribution.Gamma(1, 1)                                      # l ~ Gamma(1,1)
s = cuqi.distribution.Gamma(0, 1e-2)                                   # s ~ Gamma(0,1e-2)
x = cuqi.distribution.Gaussian(np.zeros(A.domain_dim), lambda l: 1/l)  # x ~ N(0,1/l)
y = cuqi.distribution.Gaussian(A(x), lambda s: 1/s)                    # y ~ N(A(x), 1/s)
d = cuqi.distribution.Gaussian(B(x), 0.01**2)                          # d ~ N(B(x), 0.5^2)

# %% Joint distribution
joint3 = cuqi.distribution.JointDistribution(l, s, x, y, d)    # p(l,s,x,y,d)
print(joint3)

# %% Posterior distribution
posterior3 = joint3(y=y_obs, d=d_obs)                         # p(l,s,x|y=y_obs, d=d_obs)
print(posterior3)
# %%
