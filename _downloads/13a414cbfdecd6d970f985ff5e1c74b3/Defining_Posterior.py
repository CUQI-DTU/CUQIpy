"""
How to define a Posterior distribution
======================================

The recommended way to define a posterior distribution in CUQIpy is to use the
:class:`~cuqi.distribution.JointDistribution` class to define the joint
distribution of the parameters and the data and then condition on observed
data to obtain the posterior distribution as shown in the examples below.
"""

# %%
import cuqi
import numpy as np

# %%
# A simple Bayesian inverse problem
# ---------------------------------
# 
# Consider a deconvolution inverse problem given by
#
# .. math::
#    \mathbf{y} = \mathbf{A}\mathbf{x}.
#
# See :class:`~cuqi.testproblem.Deconvolution1D` for more details.

A, y_obs, _ = cuqi.testproblem.Deconvolution1D().get_components()

# %%
# Then consider the following Bayesian model
#
# .. math::
#    \begin{align*}
#    \mathbf{x} &\sim \mathcal{N}(\mathbf{0}, 0.1\,\mathbf{I})\\
#    \mathbf{y} &\sim \mathcal{N}(\mathbf{A}\mathbf{x}, 0.05^2\,\mathbf{I})
#    \end{align*}
#
# which can be written in CUQIpy as

x = cuqi.distribution.Gaussian(np.zeros(A.domain_dim), 0.1)
y = cuqi.distribution.Gaussian(A(x), 0.05**2)

# %%
# The joint distribution :math:`p(\mathbf{x}, \mathbf{y})` is then obtained by

joint = cuqi.distribution.JointDistribution(x, y)
print(joint)

# %% 
# The posterior :math:`p(\mathbf{x}|\mathbf{y}=\mathbf{y}^\mathrm{obs})`
# is obtained by conditioning on the observed data as follows.

posterior = joint(y=y_obs)
print(posterior)

# %%
# Evaluating the posterior log density is then as simple as

posterior.logd(np.ones(A.domain_dim))

# %%
# Posterior with two forward models
# ---------------------------------
#
# Suppose we had two forward models :math:`\mathbf{A}` and :math:`\mathbf{B}`:
#
# .. math::
#    \begin{align*}
#    \mathbf{y} &= \mathbf{A}\mathbf{x}\\
#    \mathbf{d} &= \mathbf{B}\mathbf{x}\\
#    \end{align*}

# Both observations come from the same unknown x
A, y_obs, _ = cuqi.testproblem.Deconvolution1D().get_components()
B, d_obs, _ = cuqi.testproblem.Deconvolution1D(PSF="Defocus", noise_std=0.02).get_components()

# %%
# Then consider the following Bayesian model
#
# .. math::
#    \begin{align*}
#    \mathbf{x} &\sim \mathcal{N}(\mathbf{0}, 0.1\,\mathbf{I})\\
#    \mathbf{y} &\sim \mathcal{N}(\mathbf{A}\mathbf{x}, 0.05^2\mathbf{I})\\
#    \mathbf{d} &\sim \mathcal{N}(\mathbf{B}\mathbf{x}, 0.01^2\mathbf{I})
#    \end{align*}

x = cuqi.distribution.Gaussian(np.zeros(A.domain_dim), 0.1)
y = cuqi.distribution.Gaussian(A(x), 0.05**2)
d = cuqi.distribution.Gaussian(B(x), 0.01**2)

# %%
# The joint distribution :math:`p(\mathbf{x}, \mathbf{y}, \mathbf{d})` is then
# obtained by

joint2 = cuqi.distribution.JointDistribution(x, y, d)
print(joint2)

# %%
# The posterior :math:`p(\mathbf{x}|\mathbf{y}=\mathbf{y}^\mathrm{obs},\mathbf{d}=\mathbf{d}^\mathrm{obs})`
# is obtained by conditioning on the observed data as follows.

posterior2 = joint2(y=y_obs, d=d_obs)
print(posterior2)

# %%
# Evaluating the posterior log density is then as simple as

posterior2.logd(np.ones(A.domain_dim))

# %%
# Arbitrarily complex posterior distributions
# -------------------------------------------
#
# The :class:`~cuqi.distribution.JointDistribution` class can be used to
# construct arbitrarily complex posterior distributions. For example suppose
# we have the following 3 forward models
#
# .. math::
#    \begin{align*}
#    \mathbf{y} &= \mathbf{A}\mathbf{x}\\
#    \mathbf{d} &= \mathbf{B}\mathbf{x}\\
#    \mathbf{b} &= C(\mathbf{x})
#    \end{align*}
#
# where :math:`C` is a nonlinear function.

# Same x for all 3 observations
A, y_obs, _ = cuqi.testproblem.Deconvolution1D().get_components()
B, d_obs, _ = cuqi.testproblem.Deconvolution1D(PSF="Defocus", noise_std=0.02).get_components()
C = cuqi.model.Model(lambda x: np.linalg.norm(x)**2, 1, A.domain_dim)
b_obs = 16

# %%
# Then consider the following Bayesian model
#
# .. math::
#    \begin{align*}
#    q          &\sim \mathcal{U}(0.1, 10)\\
#    l          &\sim \mathrm{Gamma}(1, 1)\\
#    s          &\sim \mathrm{Gamma}(1, 10^{-2})\\
#    \mathbf{x} &\sim \mathcal{N}(\mathbf{0}, l^{-1}\mathbf{I})\\
#    \mathbf{y} &\sim \mathcal{N}(\mathbf{A}\mathbf{x}, s^{-1}\mathbf{I})\\
#    \mathbf{d} &\sim \mathcal{N}(\mathbf{B}\mathbf{x}, 0.01\mathbf{I})\\
#    \mathbf{b} &\sim \mathcal{L}(\mathbf{C}(\mathbf{x}), q)
#    \end{align*}

q = cuqi.distribution.Uniform(0.1, 10)
l = cuqi.distribution.Gamma(1, 1)
s = cuqi.distribution.Gamma(1, 1e-2)
x = cuqi.distribution.Gaussian(np.zeros(A.domain_dim), lambda l: 1/l)
y = cuqi.distribution.Gaussian(A(x), lambda s: 1/s)
d = cuqi.distribution.Gaussian(B(x), 0.01**2)
b = cuqi.distribution.Laplace(C(x), lambda q: q)

# %%
# The joint distribution :math:`p(q, l, s, \mathbf{x}, \mathbf{y}, \mathbf{d}, \mathbf{b})`
# is then obtained by

joint3 = cuqi.distribution.JointDistribution(q, l, s, x, y, d, b)
print(joint3)

# %%
# The posterior :math:`p(q, l, s, \mathbf{x}|\mathbf{y}=\mathbf{y}^\mathrm{obs},\mathbf{d}=\mathbf{d}^\mathrm{obs},\mathbf{b}=\mathbf{b}^\mathrm{obs})`
# is obtained by conditioning on the observed data as follows.

posterior3 = joint3(y=y_obs, d=d_obs, b=b_obs)
print(posterior3)

# %%
# Evaluating the posterior log density jointly over p, l, s, and :math:`\mathbf{x}`
# is then as simple as

posterior3.logd(q=1, l=1, s=1, x=np.ones(A.domain_dim))

# %%
