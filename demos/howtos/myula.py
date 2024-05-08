"""
How to sample with MYULA
======================================

The recommended way to define a posterior distribution in CUQIpy is to use the
:class:`~cuqi.distribution.JointDistribution` class to define the joint
distribution of the parameters and the data and then condition on observed
data to obtain the posterior distribution as shown in the examples below.
"""

# %%
import cuqi
import numpy as np
from cuqi.implicitprior import DenoiseRegularizer
from cuqi.experimental.mcmc import MYULANew
from cuqi.distribution import Posterior
import matplotlib.pyplot as plt

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

A, y_obs, info=cuqi.testproblem.Deconvolution1D().get_components()
# %%
# Principles behind MYULA
# ----------------------
# The goal is to solve this inverse problem by sampling from the posterior distribution given by :math:`\pi(x|y) \propto \pi(x) \pi(y|x)`.
# We assume a Gaussian likelihood, ie :math:`- \log \pi(y|x) = \|Ax-y \|_2^2/2 \texttt{sigma2}` and a prior such that :math:`- \log \pi (x) =  g(x)` with :math:`g` convex.
# To sample from :math:`\pi(x|y)`, we are going to apply a ULA based algorithm, MYULA (https://arxiv.org/pdf/1612.07471).
# We recall that ULA
#
# .. math::
#       x_{k+1} = x_k + \texttt{scale} \nabla \log \pi(x_k |y) + \sqrt{2 \texttt{scale}} z_{k+1}
#
# .. math::
#       x_{k+1} = x_k + \texttt{scale} \nabla \log \pi(y | x_k) + \texttt{scale} \nabla \log \pi(x_k) + \sqrt{2 \texttt{scale}} z_{k+1}
#
# with :math:`(z_k)_{k \in \mathbb{N}^*}` a sequence of independent and identically distributed Gaussian random variables
# with zero mean and identity covariance.
#
# In the case where :math:`\log \pi(x)` is not differentiable we can unfortunately not apply ULA. The idea is to consider a surrogate
# posterior density :math:`\pi_{\texttt{strength_smooth}} (x|y) \propto \pi(y|x) \pi_{\texttt{strength_smooth}} (x)` where
#
# .. math::
#       \pi_{\texttt{strength_smooth}}(x) \propto \exp(- g_{\texttt{strength_smooth}} (x))
#
# and :math:`g_{\texttt{strength_smooth}}` is the
# :math:`\texttt{strength_smooth}`-Moreau envelope of :math:`g`, ie
#
# .. math::
#       g_\texttt{strength_smooth}(x) = \operatorname{inf}_z \| x- z \|_2^2/2\texttt{strength_smooth} + g(z).
#
# :math:`g_{\texttt{strength_smooth}}` is continuously differentiable with :math:`1/\texttt{strength_smooth}`-Lipschitz gradient and s.t
#
# .. math::
#       \nabla g_{\texttt{strength_smooth}} (x) = (x- \operatorname{prox}_g^{\texttt{strength_smooth}} (x))/\texttt{strength_smooth}
#
# with
#
# .. math::
#       \operatorname{prox}_g^{\texttt{strength_smooth}} (x) = \operatorname{argmin}_z \|x-z \|_2^2/2\texttt{strength_smooth} + g(z)
#
# See https://link.springer.com/chapter/10.1007/978-3-319-48311-5_31 for more details.
#
# Consequently, MYULA reads as follows
#
# .. math::
#       \begin{align*}
#       x_{k+1} &= x_k + \texttt{scale} \nabla \log \pi_{\texttt{strength_smooth}}(x_k |y) + \sqrt{2 \texttt{scale}} z_{k+1}\\
#       &= x_k + \texttt{scale} \nabla \log \pi(y | x_k) + \texttt{scale} \nabla \log \pi_{\texttt{strength_smooth}}(x_k) + \sqrt{2 \texttt{scale}} z_{k+1}\\
#       &= x_k + \texttt{scale} \nabla \log \pi(y | x_k) - \texttt{scale} (x_k - \operatorname{prox}_g^{\texttt{strength_smooth}} (x))/{\texttt{strength_smooth}} + \sqrt{2 \texttt{scale}} z_{k+1}.
#       \end{align*}
#
# where :math:`\texttt{strength_smooth}` corresponds to the smoothing strength of :math:`g`.
#
# To illustrate MYULA, we will consider :math:`g(x) = \texttt{strength_reg} \  TV(x) = \texttt{strength_reg} \|\nabla x \|_{2, 1}`,
# where :math:`\texttt{strength_reg}` is the regularization parameter which controls the regularization strength induced by TV.

# %%
# Bayesian model definition
# -------------------------
# Then consider the following Bayesian model
#
# .. math::
#    \begin{align*}
#    \mathbf{x} &\sim \exp (- \texttt{strength_reg} \|\nabla x \|_{2,1})\\
#    \mathbf{y} &\sim \mathcal{N}(\mathbf{A}\mathbf{x}, \texttt{sigma2}\,\mathbf{I}) \ ,
#    \end{align*}
#
# with :math:`\texttt{sigma2}=0.05^2`.

# %%
# Likelihood definition
# ---------------------
# We first specify the data distribution as follows:
sigma2=0.05**2
y=cuqi.distribution.Gaussian(A, sigma2)
# %%
# Then we can define the likelihood with
likelihood=y(y=y_obs)

# %%
# Implicit prior definition
# ---------------------------------
# To apply MYULA, we need to define the implicit prior :math:`\pi_{\texttt{stength_smooth}}(x)`. Evaluating this surrogate prior is doable but too intensive from
# a computational point of view as it requires to solve an optimization problem. However to apply MYULA, we only require access to
# :math:`\operatorname{prox}_{\texttt{strength_reg}\ TV}^{\texttt{strength_smooth}}`.
#
# As suggested by Durmus et al. (https://arxiv.org/pdf/1612.07471), we set the smoothing parameter :math:`\texttt{strength_smooth} \approx \texttt{sigma2}`, ie :math:`\texttt{strength_smooth}= 0.5 \ \texttt{sigma2}`.
#
# We set the regularization parameter to :math:`\texttt{stength_reg}=10`.

# %%
# Regularization and smoothing parameters definition
strength_reg=10
strength_smooth=0.5*sigma2
# %%
# To estimate :math:`\operatorname{prox}_{\texttt{strength_reg}\  TV}^{\texttt{strength_smooth}}` we use the implementation provided by Scikit-Image. But we can use any solver to compute this quantity.
#
# We emphasize that we have for any :math:`g`
#
# .. math::
#       \operatorname{prox}_{\texttt{strength_reg}\  g}^{\texttt{strength_smooth}} = \operatorname{prox}_{g}^{\texttt{weight}} ,
#
# with :math:`\texttt{weight} = \texttt{strength_reg} \times  \texttt{strength_smooth}`.
from skimage.restoration import denoise_tv_chambolle
def prox_g(x, strength_reg=None, strength_smooth=None):
    weight=strength_reg*strength_smooth
    return denoise_tv_chambolle(x, weight=weight, max_num_iter=100), True
# %%
# We save all the important variables into the variable :math:`\texttt{denoiser_setup}`.
denoiser_setup={}
denoiser_setup["strength_reg"]=strength_reg
denoiser_setup["strength_smooth"]=strength_smooth
# %%
# Now we can define our implicit prior.
denoise_regularizer = DenoiseRegularizer(
    prox_g,
    strength_smooth=strength_smooth,
    denoiser_setup=denoiser_setup,
    geometry=likelihood.model.domain_geometry
)

# %%
# Implicitly defined posterior distribution
# -----------------------------------------
# We can now define the implicitly defined posterior distribution as follows:
posterior=Posterior(likelihood, denoise_regularizer)

# %%
# Parameters of the MYULA sampler
# ------------------------------
# We let run MYULA for :math:`\texttt{Ns}=10^4`
# iterations. We discard the :math:`\texttt{Nb}=1000` first burn-in samples of the Markov chain. Furthermore, as MCMC methods generate
# correlated samples, we also perform a thinning: we only consider 1 samples every :math:`\texttt{Nt}=20`
# samples to compute our quantities of interest.
# :math:`\texttt{scale}` is set wrt the recommendation of Durmus et al. (https://arxiv.org/pdf/1612.07471).
Ns=10000
Nb=1000
Nt=20
# Step-size of MYULA
scale=0.9/(1/sigma2 + 1/strength_smooth)
# %%
# In order to get reproducible results, we set the seed parameter to 0.
np.random.seed(0)
# %%
# MYULA sampler
# -------------
# Definition of the MYULA sampler.
myula_sampler=MYULANew(target=posterior, scale=scale)
# %%
# Sampling with MYULA.
myula_sampler.sample(Ns=Ns)
# %%
# Retrieve the samples. We apply the burnin and perform thinning to the Markov chain.
samples=myula_sampler.get_samples()
samples_warm=samples.burnthin(Nb=Nb, Nt=Nt)
# %%
# Results
# -------
# Mean and CI plot.
plt.figure(figsize=(10, 10))
y_obs.plot(label="Observation")
samples_warm.plot_ci(exact=info.exactSolution)
plt.legend()
# %%
# Standard  deviation plot to estimate the uncertainty.
plt.figure(figsize=(10, 10))
samples_warm.plot_std()
# %%
# Samples autocorrelation plot.
samples_warm.plot_autocorrelation(max_lag=100)
