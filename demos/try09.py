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
cov = TP.likelihood.cov
n = model.domain_geometry.dim
m = model.range_geometry.dim
x_true = TP.exactSolution

# %%
# Define Gaussian likelihood and prior
likelihood = cuqi.distribution.GaussianCov(model, cov)

var = 0.2
prior = cuqi.distribution.GaussianCov(0, var*np.ones(n))

# %%
# Define potential of posterior (returns logpdf and gradient w.r.t x)

def posterior_logpdf(x):
    logpdf = -prior.logpdf(x) - likelihood(x=x).logpdf(data) 
    return logpdf

def potential(x):
    logpdf = posterior_logpdf(x) 
    grad = -prior.gradient(x) - likelihood.gradient(data,x=x)
    return logpdf, grad

# %%
# Compare with MAP computed using BayesianProblem (within the testproblem)
x0 = np.random.randn(n)

# Exact  MAP
prior2 = cuqi.distribution.Gaussian(np.zeros(n), np.sqrt(var), np.eye(n))
TP.prior = prior2
x_MAP_exact = TP.MAP()
print('relative error exact MAP:', np.linalg.norm(x_MAP_exact-x_true)/np.linalg.norm(x_true))

# L_BFGS_B MAP
# Solve posterior problem using L_BFGS_B
solver = cuqi.solver.L_BFGS_B(potential, x0)
x_MAP_LBFGS = solver.solve()
print('relative error L-BFGS MAP:', np.linalg.norm(x_MAP_LBFGS-x_true)/np.linalg.norm(x_true))

#%% BFGS MAP
# Solve posterior problem using L_BFGS_B
solver = cuqi.solver.minimize(posterior_logpdf, x0)
x_MAP_BFGS = solver.solve()
print('relative error BFGS MAP:', np.linalg.norm(x_MAP_BFGS-x_true)/np.linalg.norm(x_true))


# %%
# plots
plt.plot(x_true, 'k-', label = "True")
plt.plot(x_MAP_exact, 'b-', label = "Exact MAP")
plt.plot(x_MAP_LBFGS, 'r--', label  = "LBFGS MAP")
plt.plot(x_MAP_BFGS, 'r--', label  = "BFGS MAP")
plt.legend()
plt.show()
# %%
