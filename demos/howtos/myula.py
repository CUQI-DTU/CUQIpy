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
#
# The goal is to solve this inverse problem by sampling from the posterior distribution given by $\pi(x|y) \propto \pi(x) \pi(y|x)$.
# We assume a Gaussian likelihood, ie $- \log \pi(y|x) = \|Ax-y \|_2^2/2\sigma^2$ and a prior such that $- \log \pi (x) =  g(x)$ with $g$ convex.
# To sample from $\pi(x|y)$, we are going to apply a ULA based algorithm, MYULA (https://arxiv.org/pdf/1612.07471).
# We recall that ULA  
# $$ x_{k+1} = x_k + \delta \nabla \log \pi(x_k |y) + \sqrt{2 \delta} z_{k+1} \ , $$
# $$ x_{k+1} = x_k + \delta \nabla \log \pi(y | x_k) + \delta \nabla \log \pi(x_k) + \sqrt{2 \delta} z_{k+1} \ ,$$
# with $(z_k)_{k \in \mathbb{N}^*}$ a sequence of independent and identically distributed Gaussian random variables
# with zero mean and identity covariance.
# In the case where $\log \pi(x)$ is not differentiable we can unfortunately not apply ULA. The idea is to consider a surrogate 
# posterior density $\pi_\alpha (x|y) \propto \pi(y|x) \pi_\alpha (x)$ where $\pi_\alpha(x) \propto \exp(- g_\alpha (x))$ and $g_\alpha$ is the
# $\alpha$-Moreau envelope of $g$, ie $g_\alpha(x) \operatorname{inf}_z \| x- z \|_2^2/2\alpha + g(z)$.
# $g_\alpha$ is continuously differentiable with $1/\alpha$-Lipschitz gradient and s.t $\nabla g_\alpha (x) = (x- \operatorname{prox}_g^\alpha (x))/\alpha$
# with $prox_g^\alpha (x) = \operatorname{argmin}_z \|x-z \|_2^2/2\alpha + g(z) $ (see https://link.springer.com/chapter/10.1007/978-3-319-48311-5_31).
# Consequently, MYULA reads as follows
# $$ x_{k+1} = x_k + \delta \nabla \log \pi_\alpha(x_k |y) + \sqrt{2 \delta} z_{k+1} \ , $$
# $$ x_{k+1} = x_k + \delta \nabla \log \pi(y | x_k) + \delta \nabla \log \pi_\alpha(x_k) + \sqrt{2 \delta} z_{k+1} \ ,$$
# $$ x_{k+1} = x_k + \delta \nabla \log \pi(y | x_k) - \delta (x_k - \operatorname{prox}_g^\alpha (x))/\alpha + \sqrt{2 \delta} z_{k+1} \ .$$
# $\alpha$ corresponds to the smoothing strength of $g$.
# To illustrate MYULA, we will consider the $g(x) = \gamma TV(x) = \gamma \|\nabla x \|_{2, 1}$,
# where $\gamma$ is the regularization parameter which controls the regularization strength induced by TV.  

A, y_obs, info = cuqi.testproblem.Deconvolution1D().get_components()

# %%
# Then consider the following Bayesian model
#
# .. math::
#    \begin{align*}
#    \mathbf{x} &\sim \exp (-\gamma \|\nabla x \|_{2,1}\\
#    \mathbf{y} &\sim \mathcal{N}(\mathbf{A}\mathbf{x}, 0.05^2\,\mathbf{I})
#    \end{align*}
#
# In CUQI we an easily specify the likelihood as follows:

sigma2 = 0.05**2
y = cuqi.distribution.Gaussian(A, sigma2)
likelihood = y(y = y_obs)
from skimage.metrics import mean_squared_error as mse  
print("MSE(Ax, y) = ", mse(info.exactData, y_obs))

# %%
# To apply MYULA, we need to define the implicit prior $\pi_\alpha(x)$. Evaluating this surrogate prior is doable but too intensive from
# a computational point of view as it requires to solve an optimization problem. However to apply MYULA, we only require access to 
# the proximal operator of $g$ ie $\gamma TV$. 
# As suggested by Durmus et al. (https://arxiv.org/pdf/1612.07471), we set the smoothing parameter $\alpha \approx \sigma^2 $, ie $\texttt{strength_smooth}= 0.5*sigma2$.
# We set the regularization parameter to $\texttt{stength_reg}=10$.
from skimage.restoration import denoise_tv_chambolle
strength_reg = 10
strength_smooth = 0.5*sigma2#np.copy(sigma2)
weight = strength_reg*strength_smooth
def prox_g(x):
    return denoise_tv_chambolle(x, weight = weight, max_num_iter = 100), True
denoise_regularizer = DenoiseRegularizer(prox_g, strength_reg = strength_smooth)

# %%
# In this section we define the important parameters for MYULA. We let run MYULA for $\texttt{Ns}=10^4$
# iterations. We discard the $\texttt{Nb}=1000$ first samples of the Markov chain, as they are not
# sampled from the stationary distribution of the Markov chain. Furthermore, as MCMC methods generate
# correlated samples, we also perform a thinning, ie we only consider 1 samples every $\texttt{Nt}=20$
# samples to compute our quantities of interest.
# The scale parameter $\delta$ is set wrt the recommendation of Durmus et al. (https://arxiv.org/pdf/1612.07471).
Ns = 10000
Nb = 1000
Nt = 20
# Step-size of MYULA
scale = 0.9/(1/sigma2 + 1/strength_smooth)
# %%
# Definition of the MYULA sampler and run of the chain
myula_sampler = MYULANew(likelihood = likelihood, denoise_regularizer = denoise_regularizer, scale = scale)
myula_sampler.sample(Ns = Ns)
samples = myula_sampler.get_samples()
samples_warm = samples.burnthin(Nb = Nb, Nt = Nt)
#%%
# Results
plt.figure(figsize = (10, 10))
samples_warm.plot_mean(label = "MYULA mean")
info.exactSolution.plot(label = "Exact solution")
y_obs.plot(label = "Observation")
samples_warm.plot_ci()
plt.legend()

plt.figure(figsize = (10, 10))
samples_warm.plot_std()

plt.figure(figsize = (10, 10))
samples_warm.plot_autocorrelation(max_lag = 300)




