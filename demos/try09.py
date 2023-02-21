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
TP = cuqi.testproblem._Deblur()
model = TP.model #Deblur model
data = TP.data #Data from deblur problem
cov = TP.likelihood.distribution.cov
n = model.domain_dim
m = model.range_dim
x_true = TP.exactSolution

# %%
# Define Gaussian likelihood and prior
likelihood = cuqi.distribution.Gaussian(model, cov, name="y").to_likelihood(data)

var = 0.2
prior = cuqi.distribution.Gaussian(0, var*np.ones(n), name="x")

# %% MAP estimates
# Define potential of posterior (returns logpdf and gradient w.r.t x)

def posterior_logpdf(x):
    logpdf = -prior.logd(x) - likelihood.logd(x)
    return logpdf

def posterior_logpdf_grad(x):
    grad = -prior.gradient(x) - likelihood.gradient(x) 
    return grad

# Starting point
x0 = np.random.randn(n)

# Exact  MAP
prior2 = cuqi.distribution.Gaussian(np.zeros(n), np.sqrt(var), np.eye(n))
TP.prior = prior2
x_MAP_exact = TP.MAP()
print('relative error exact MAP:', np.linalg.norm(x_MAP_exact-x_true)/np.linalg.norm(x_true))

#%% L_BFGS_B MAP
solver = cuqi.solver.L_BFGS_B(posterior_logpdf, x0, gradfunc = posterior_logpdf_grad)
x_MAP_LBFGS, info_MAP_LBFGS = solver.solve()
print('relative error L-BFGS MAP:', np.linalg.norm(x_MAP_LBFGS-x_true)/np.linalg.norm(x_true))

#%% L_BFGS_B MAP using minimize
solver = cuqi.solver.minimize(posterior_logpdf, x0, gradfunc = posterior_logpdf_grad, method = "L-BFGS-B")
x_MAP_LBFGS2, info_MAP_LBFGS2 = solver.solve()
print('relative error L-BFGS MAP:', np.linalg.norm(x_MAP_LBFGS2-x_true)/np.linalg.norm(x_true))

#%% BFGS MAP without gradient input
solver = cuqi.solver.minimize(posterior_logpdf, x0)
x_MAP_BFGS, info_MAP_BFGS = solver.solve()
print('relative error BFGS MAP:', np.linalg.norm(x_MAP_BFGS-x_true)/np.linalg.norm(x_true))

#%% BFGS MAP with gradient input
solver = cuqi.solver.minimize(posterior_logpdf, x0, gradfunc = posterior_logpdf_grad)
x_MAP_BFGSgrad, info_MAP_BFGSgrad = solver.solve()
print('relative error BFGS MAP:', np.linalg.norm(x_MAP_BFGSgrad-x_true)/np.linalg.norm(x_true))


#%% SLSQP MAP
solver = cuqi.solver.minimize(posterior_logpdf, x0, method = 'SLSQP')
x_MAP_SLSQP, info_MAP_SLSQP = solver.solve()
print('relative error SLSQP MAP:', np.linalg.norm(x_MAP_SLSQP-x_true)/np.linalg.norm(x_true))


# %% plots
plt.plot(x_true, 'k-', label = "True")
plt.plot(x_MAP_exact, 'b-', label = "Exact MAP")
plt.plot(x_MAP_LBFGS, 'r--', label  = "LBFGS MAP")
plt.plot(x_MAP_BFGS, 'y:', label  = "BFGS MAP, no grad func")
plt.plot(x_MAP_BFGSgrad, 'm:', label  = "BFGS MAP, with grad func")
plt.plot(x_MAP_SLSQP, 'g:', label  = "SLSQP MAP")
plt.legend()
plt.show()

#%% ML estimates

def likelihood_logpdf(x):
    logpdf = - likelihood.logd(x)
    return logpdf

def likelihood_logpdf_grad(x):
    grad =  - likelihood.gradient(x)
    return grad

# L_BFGS_B MAP
solver = cuqi.solver.L_BFGS_B(likelihood_logpdf, x0, gradfunc = likelihood_logpdf_grad)
x_ML_LBFGS, info_ML_LBFGS = solver.solve()
print('relative error L-BFGS ML:', np.linalg.norm(x_MAP_LBFGS-x_true)/np.linalg.norm(x_true))

# BFGS MAP
solver = cuqi.solver.minimize(likelihood_logpdf, x0)
x_ML_BFGS, info_ML_BFGS = solver.solve()
print('relative error BFGS ML:', np.linalg.norm(x_MAP_BFGS-x_true)/np.linalg.norm(x_true))

# %% plots
plt.plot(x_true, 'k-', label = "True")
#plt.plot(x_ML_exact, 'b-', label = "Exact ML")
plt.plot(x_ML_LBFGS, 'r--', label  = "LBFGS ML")
plt.plot(x_ML_BFGS, 'y:', label  = "BFGS ML")
plt.legend()
plt.show()


# %%

prob = cuqi.problem.BayesianProblem(likelihood, prior)
# %%
MAP_prob = prob.MAP()
# %%
print('relative error BFGS MAP:', np.linalg.norm(MAP_prob-x_true)/np.linalg.norm(x_true))
# %%
print('relative error BFGS MAP:', np.linalg.norm(MAP_prob-x_MAP_BFGS)/np.linalg.norm(x_MAP_BFGS))
# %%

# %%
ML_prob = prob.ML()
# %%
print('relative error BFGS ML:', np.linalg.norm(ML_prob-x_true)/np.linalg.norm(x_true))
# %%
print('relative error BFGS ML:', np.linalg.norm(ML_prob-x_ML_BFGS)/np.linalg.norm(x_MAP_BFGS))

# %%
