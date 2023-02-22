#%% This file show-cases the Deconvolution 1D test problem.
import sys
sys.path.append("..")
import numpy as np
import scipy as sp
import scipy.stats as sps
import matplotlib.pyplot as plt

from cuqi.testproblem import Deconvolution1D
from cuqi.distribution import Gaussian, InverseGamma, JointDistribution, GMRF
from cuqi.sampler import Gibbs, Linear_RTO, Conjugate 

# ===================================================================
# deconvolution problem
# ===================================================================
# Model and data
A, y_data, probinfo = Deconvolution1D.get_components(noise_type='normalizedgaussian', \
                    kernel_param=55, phantom='square', noise_std=0.05)
y_true = probinfo.exactData
m = len(y_data)

# Get dimension of signal
d = A.domain_dim
grid = A.domain_geometry.grid
f_true = probinfo.exactSolution   # true signal
# sigma_obs = np.sqrt(likelihood.distribution.cov)
# lambd_obs = 1/(sigma_obs**2)

# 1D finite difference matrix with zero BCs
D = sp.sparse.spdiags([-np.ones(d), np.ones(d)], [-1, 0], d+1, d).tocsc()
D = D[:-1, :]

plt.figure()
plt.plot(grid, f_true, 'b-')
plt.plot(grid, y_data, 'r*')
plt.plot(grid, y_true, 'k-')
plt.show()

# ===================================================================
# Define 'hierarchical' likelihood
s2 = InverseGamma(1, 0, 1e-4)    # noise variance
y = Gaussian(A, lambda s2: s2)  # data distribution: likelihood

# ===================================================================
# Define extended horseshoe hierarchical prior
nu = 1   # half-Cauchy hyperpriors of the original horseshoe

# prior hyperparams
g = InverseGamma(1/2, 0, 1)                                 # auxiliary param 1
xi = InverseGamma((1/2)*np.ones(d), 0, 1)                   # auxiliary param 2
t2 = InverseGamma(nu/2, 0, lambda g: nu/g)                  # global variance param
w2 = InverseGamma((nu/2)*np.ones(d), 0, lambda xi: nu/xi)   # local variance param

# conditionally Gaussian
x = Gaussian(np.zeros(d), sqrtprec=lambda t2, w2: sp.sparse.diags(1/np.sqrt(t2*w2), format='csc') @ D)           

# set initial states
x.init_point = 0.5*np.ones(d)
s2.init_point = 1e-2
t2.init_point = 1
w2.init_point = abs(sps.t.rvs(2, scale=1, size=d))
g.init_point = 1
xi.init_point = np.ones(d)

# ===================================================================
# Combine into a joint posterior distribution
joint = JointDistribution(x, s2, t2, w2, g, xi, y)
posterior = joint(y=y_data)

# View the joint distribution
print(joint)

# Define sampling strategy
sampling_strategy = {
    'x': Conjugate,
    's2': Conjugate,
    't2': Conjugate,
    'w2': Conjugate,
    'g': Conjugate,
    'xi': Conjugate
}

# check Gibbs HS implementation
# np.random.seed(1)
# conditional = posterior(s2=1e-3, t2=1e-4, w2=np.ones(d), g=0.1, xi=np.ones(d))
# print(conditional)
# sample = Conjugate(conditional).step(np.ones(d))
# print(sample)

# Define Gibbs sampler
sampler = Gibbs(posterior, sampling_strategy, True)

# Run sampler
samples = sampler.sample(Ns=5000, Nb=2000) # 400000
# samples.BurnThin(0, 2)

# sample_gibbs_step2000
# np.random.seed(1)
# conditional_x = post(s2=..., ...)
# sample = Conjugate(conditional_x).step(x0)

# Plot credible intervals for the signal
plt.figure()
samples['x'].plot_ci(exact=probinfo.exactSolution)

plt.figure()
samples['w2'].plot_ci()

# Trace plot for d
samples['t2'].plot_trace(figsize=(8,2))
# tau2_sqrt = cuqi.model.Model(lambda x: np.sqrt(x), 1, 1)
# samples2 = tau2_sqrt(samples['t2']).plot_trace(figsize=(8,2))

plt.show()
