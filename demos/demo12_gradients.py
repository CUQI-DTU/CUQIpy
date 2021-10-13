# %%
import sys
sys.path.append("../")
import time
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.sparse as sps

# myfuns
import cuqi

# %%
# Load cuqi deblur model and data
TP = cuqi.testproblem.Deblur()
model = TP.model #Deblur model
data = TP.data #Data from deblur problem
sigma2 = TP.likelihood.std**2
n = model.domain_geometry.dim
m = model.range_geometry.dim
x_true = TP.exactSolution

# %%
# Define Gaussian likelihood and prior
likelihood = cuqi.distribution.GaussianCov(model, sigma2)
pr = 'cauchy'
if (pr == 'gaussian'):
    var = 0.2
    prior = cuqi.distribution.GaussianCov(0, var*np.ones(n))
elif (pr == 'cauchy'):
    h = 1/n
    delta = 0.3
    prior = cuqi.distribution.Cauchy_diff(np.zeros(n), delta*h, 'neumann')

# %%
# Define potential of posterior (returns logpdf and gradient w.r.t x)
def potential(x):
    logpdf = -prior.logpdf(x) - likelihood(x=x).logpdf(data) 
    grad = -prior.gradient(x) - likelihood.gradient(data,x=x)
    return logpdf, grad

# %%
# Compare with MAP computed using BayesianProblem (within the testproblem)
x0 = np.random.randn(n)
if (pr == 'gaussian'):
    prior2 = cuqi.distribution.Gaussian(np.zeros(n), np.sqrt(var), np.eye(n))
    TP.prior = prior2
    x_MAP = TP.MAP()
else:
    # Solve posterior problem using BFGS
    solver = cuqi.solver.L_BFGS_B(potential, x0)
    x_MAP = solver.solve()
print('relative error MAP:', np.linalg.norm(x_MAP-x_true)/np.linalg.norm(x_true))

# %%
# sampling using NUTS
MCMC = cuqi.sampler.NUTS(likelihood, prior, data, x0, 12)
Ns = int(1e3)      # number of samples
Nb = int(0.5*Ns)   # burn-in
samples = MCMC.sample(Ns, Nb)
#
xs = samples[0]
x_mean = np.mean(xs, axis=1)
x_std = np.std(xs, axis=1)
print('relative error mean:', np.linalg.norm(x_mean-x_true)/np.linalg.norm(x_true))

# %%
# plots
model.domain_geometry.plot(x_true, 'k-')
model.domain_geometry.plot(x_MAP, 'b-')
# model.domain_geometry.plot(x_MAP, 'r--')
model.domain_geometry.plot(x_mean, 'r--')
model.domain_geometry.plot(x_mean+x_std, 'm--')
model.domain_geometry.plot(x_mean-x_std, 'm--')
plt.show()