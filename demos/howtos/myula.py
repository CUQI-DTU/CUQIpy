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
from cuqi.implicitprior import RestorationPrior, MoreauYoshidaPrior
from cuqi.sampler import ULA, MYULA
from cuqi.distribution import Posterior
import matplotlib.pyplot as plt

from skimage.metrics import normalized_root_mse as nrmse
from skimage.metrics import mean_squared_error as mse

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

A, y_obs, info = cuqi.testproblem.Deconvolution1D().get_components()
# %%
# Principles behind MYULA
# ----------------------
# The goal is to solve this inverse problem by sampling from the posterior
# distribution given by :math:`\pi(x|y) \propto \pi(x) \pi(y|x)`.
# We assume a Gaussian likelihood, ie :math:`- \log \pi(y|x) = \|Ax-y \|_2^2/2 \texttt{sigma2}`
# and a prior such that :math:`- \log \pi (x) =  g(x)` with :math:`g` convex.
# To sample from :math:`\pi(x|y)`, we are going to apply a ULA based algorithm,
# MYULA (https://arxiv.org/pdf/1612.07471).
# We recall that ULA
#
# .. math::
#       x_{k+1} = x_k + \texttt{scale} \nabla \log \pi(x_k |y) + \sqrt{2 \texttt{scale}} z_{k+1}
#
# .. math::
#       x_{k+1} = x_k + \texttt{scale} \nabla \log \pi(y | x_k) + \texttt{scale} \nabla \log \pi(x_k) + \sqrt{2 \texttt{scale}} z_{k+1}
#
# with :math:`(z_k)_{k \in \mathbb{N}^*}` a sequence of independent and
# identically distributed Gaussian random variables
# with zero mean and identity covariance.
#
# In the case where :math:`\log \pi(x)` is not differentiable we can
# unfortunately not apply ULA. The idea is to consider a surrogate
# posterior density :math:`\pi_{\texttt{smoothing_strength}} (x|y) \propto \pi(y|x) \pi_{\texttt{smoothing_strength}} (x)`
# where
#
# .. math::
#       \pi_{\texttt{smoothing_strength}}(x) \propto \exp(- g_{\texttt{smoothing_strength}} (x))
#
# and :math:`g_{\texttt{smoothing_strength}}` is the
# :math:`\texttt{smoothing_strength}`-Moreau envelope of :math:`g`, ie
#
# .. math::
#       g_\texttt{smoothing_strength}(x) = \operatorname{inf}_z \| x- z \|_2^2/2\texttt{smoothing_strength} + g(z).
#
# :math:`g_{\texttt{smoothing_strength}}` is continuously differentiable with :math:`1/\texttt{smoothing_strength}`-Lipschitz gradient and s.t
#
# .. math::
#       \nabla g_{\texttt{smoothing_strength}} (x) = (x- \operatorname{prox}_g^{\texttt{smoothing_strength}} (x))/\texttt{smoothing_strength}
#
# with
#
# .. math::
#       \operatorname{prox}_g^{\texttt{smoothing_strength}} (x) = \operatorname{argmin}_z \|x-z \|_2^2/2\texttt{smoothing_strength} + g(z)
#
# See https://link.springer.com/chapter/10.1007/978-3-319-48311-5_31 for more details.
#
# MYULA consists in applying ULA to a smoothed target distribution. It reads
#
# .. math::
#       \begin{align*}
#       x_{k+1} &= x_k + \texttt{scale} \nabla \log \pi_{\texttt{smoothing_strength}}(x_k |y) + \sqrt{2 \texttt{scale}} z_{k+1}\\
#       &= x_k + \texttt{scale} \nabla \log \pi(y | x_k) + \texttt{scale} \nabla \log \pi_{\texttt{smoothing_strength}}(x_k) + \sqrt{2 \texttt{scale}} z_{k+1}\\
#       &= x_k + \texttt{scale} \nabla \log \pi(y | x_k) - \texttt{scale} (x_k - \operatorname{prox}_g^{\texttt{smoothing_strength}} (x_k))/{\texttt{smoothing_strength}} + \sqrt{2 \texttt{scale}} z_{k+1}.
#       \end{align*}
#
# where :math:`\texttt{smoothing_strength}` corresponds to the smoothing strength of :math:`g`.
#
# To illustrate MYULA, we will consider :math:`g(x) = \texttt{regularization_strength} \  TV(x) = \texttt{regularization_strength} \|\nabla x \|_{2, 1}`,
# where :math:`\texttt{regularization_strength}` is the regularization parameter which
# controls the regularization strength induced by TV.

# %%
# Bayesian model definition
# -------------------------
# Then consider the following Bayesian Inverse Problem:
#
# .. math::
#    \begin{align*}
#    \mathbf{x} &\sim \exp (- \texttt{regularization_strength} \|\nabla x \|_{2,1})\\
#    \mathbf{y}_{obs} &\sim \mathcal{N}(\mathbf{A}\mathbf{x}, \texttt{sigma2}\,\mathbf{I}) \ ,
#    \end{align*}
#
# with :math:`\texttt{sigma2}=0.05^2`.

# %%
# Likelihood definition
# ---------------------
# We first specify the data distribution as follows:
sigma2 = 0.05**2
y = cuqi.distribution.Gaussian(A, sigma2)
# %%
# Then we can define the likelihood with
likelihood = y(y=y_obs)

# %%
# RestorationPrior and MoreauYoshidaPrior
# ---------------------------------------
# To apply MYULA, we need to define the Moreau-Yoshida prior
# :math:`\pi_{\texttt{smoothing_stength}}(x)`.
# Evaluating this surrogate prior is doable but too intensive from
# a computational point of view as it requires to solve an optimization problem.
# However to apply MYULA, we only require access to
# :math:`\operatorname{prox}_{\texttt{regularization_strength}\ TV}^{\texttt{smoothing_strength}}`.
# :math:`\operatorname{prox}_{\texttt{regularization_strength}\ TV}^{\texttt{smoothing_strength}}`
# is a denoising operator (also called denoiser), which takes a signal as input
# and returns a less noisy
# signal. In CUQIPy, we talk about restoration operators (also called restorators).
# Denoisers are an example of restorators.
# Restorators are at the
# core of a specific type of priors called RestorationPrior. We cannot sample
# from these priors but they allow us to define other types of priors.

# %%
# RestorationPrior definition
# ---------------------------
# A restorator, is associated with a parameter called
# :math:`\texttt{restoration_strength}`. This parameter indicates how strong is
# the restoration. For example, when this restorator is a denoiser, an operator
# taking an signal as input and returning a less noisy signal, :math:`\texttt{restoration_strength}`
# can correspond to the  denoising level.
# In the following, we consider the denoiser
# :math:`\operatorname{prox}_{\texttt{regularization_strength}\ TV}^{\texttt{restoration_strength}}`.
# We use the implementation provided by Scikit-Image. But we can use any solver
# to compute this quantity.
# We emphasize that we have for any :math:`g`
#
# .. math::
#       \operatorname{prox}_{\texttt{regularization_strength}\  g}^{\texttt{smoothing_strength}} = \operatorname{prox}_{g}^{\texttt{weight}} ,
#
# with :math:`\texttt{weight} = \texttt{regularization_strength} \times  \texttt{smoothing_strength}`.
regularization_strength = 10
restoration_strength = 0.5 * sigma2
from skimage.restoration import denoise_tv_chambolle


def prox_g(x, regularization_strength=None, restoration_strength=None):
    weight = regularization_strength * restoration_strength
    return denoise_tv_chambolle(x, weight=weight, max_num_iter=100), None


# %%
# We save all the important variables into the variable
# :math:`\texttt{restorator_kwargs}`.
restorator_kwargs = {}
restorator_kwargs["regularization_strength"] = regularization_strength
# %%
# Now we can define our RestorationPrior.
restorator = RestorationPrior(
    prox_g,
    restorator_kwargs=restorator_kwargs,
    geometry=likelihood.model.domain_geometry,
)
# %% Illustration of the effect of the denoiser.
# We first apply the restorate method of our denoiser to :math:`\mathbf{y}_{obs}`.
# This operator should restore :math:`\mathbf{y}_{obs}` and generate a signal close
# to :math:`\mathbf{A}\mathbf{x}`.
res = restorator.restore(y_obs, restoration_strength)
# %%
# In this cell, we show the effect of the restorator both from a visual
# and quantitative point of view. We use the relative error and the mean-squared
# error. The smaller are these quantities, the better it is.

plt.figure(figsize=(10, 10))
y_obs.plot(
    label="observation (Relative error={:.5f})".format(nrmse(info.exactData, y_obs))
)
res.plot(label="restoration (Relative error={:.5f})".format(nrmse(info.exactData, res)))
info.exactData.plot(label="groundtruth")
plt.legend()

print(
    "MSE(Ax, y_obs) is ",
    mse(info.exactData, y_obs) / mse(info.exactData, res),
    " times larger than MSE(Ax, res).",
)

# %%
# Definition of the Moreau-Yoshida prior
# --------------------------------------
# It is a smoothed version from the target prior. Its definition requires a prior
# of type RestorationPrior and a scalar parameter :math:`\texttt{smoothing_strength}`
# which controls the strength of the smoothing. We must have
# :math:`\texttt{smoothing_strength}=\texttt{restoration_strength}`.
#
# As suggested by Durmus et al. (https://arxiv.org/pdf/1612.07471), we set the
# smoothing parameter :math:`\texttt{smoothing_strength} \approx \texttt{sigma2}`,
# ie :math:`\texttt{smoothing_strength}= 0.5 \ \texttt{sigma2}`.
myprior = MoreauYoshidaPrior(prior=restorator, smoothing_strength=restoration_strength)

# %%
# Implicitly defined posterior distribution
# -----------------------------------------
# We can now define the implicitly defined smoothed posterior distribution as
# follows:
smoothed_posterior = Posterior(likelihood, myprior)

# %%
# Parameters of the MYULA sampler
# -------------------------------
# We let run MYULA for :math:`\texttt{Ns}=10^4`
# iterations. We discard the :math:`\texttt{Nb}=1000` first burn-in samples of
# the Markov chain. Furthermore, as MCMC methods generate
# correlated samples, we also perform a thinning: we only consider 1 samples
# every :math:`\texttt{Nt}=20`
# samples to compute our quantities of interest.
# :math:`\texttt{scale}` is set wrt the recommendation of Durmus et al.
# (https://arxiv.org/pdf/1612.07471). It must be smaller than the inverse of the
# Lipschitz constant of the gradient of the log-posterior density. In this setting,
# The Lipschitz constant of the gradient of likelihood log-density is
# :math:`\|A^TA \|_2^2/\texttt{sigma2}` and the one of the log-prior is
# :math:`1/\texttt{smoothing_strength}`.
Ns = 10000
Nb = 1000
Nt = 20
# Step-size of MYULA
scale = 0.9 / (1 / sigma2 + 1 / restoration_strength)
# %%
# In order to get reproducible results, we set the seed parameter to 0.
np.random.seed(0)
# %%
# ULA sampler
# -------------
# Definition of the ULA sampler which aims at sampling from the smoothed posterior.
ula_sampler = ULA(target=smoothed_posterior, scale=scale)
# %%
# Sampling with ULA from the smoothed target posterior.
ula_sampler.sample(Ns=Ns)
# %%
# Retrieve the samples. We apply the burnin and perform thinning to the Markov
# chain.
samples = ula_sampler.get_samples()
samples_warm = samples.burnthin(Nb=Nb, Nt=Nt)
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
# %%
# Other way to sample with MYULA
# ------------------------------
# To sample with MYULA, we can also define an implicit posterior with a
# RestorationPrior object (instead of MoreauYoshidaPrior object) and then automatically
# perform the Moreau-Yoshida smoothing when defining
# the MYULA sampler.
posterior = Posterior(likelihood, restorator)
# %%
# Definition of the MYULA sampler
# -------------------------------
# Again, we must have :math:`\texttt{smoothing_strength}=\texttt{restoration_strength}`.
myula_sampler = MYULA(
    target=posterior, scale=scale, smoothing_strength=restoration_strength
)
# %%
# We then sample using the MYULA sampler. It targets the same smoothed distribution
# as the ULA sampler applied with the smoothed posterior distribution.
# If the samples generated by ULA and MYULA are the same, then a message "MYULA
# samples of the posterior are the same ULA samples from the smoothed
# posterior"
np.random.seed(0)
myula_sampler.sample(Ns=Ns)
samples_myula = myula_sampler.get_samples()
samples_myula_warm = samples_myula.burnthin(Nb=Nb, Nt=Nt)
assert np.allclose(samples_warm.samples, samples_myula_warm.samples)
print(
    "MYULA samples of the posterior are the same ULA samples from the smoothed \
posterior"
)
