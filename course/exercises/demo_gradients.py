# %%
import sys
sys.path.append("../../")
import time
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.sparse as sps

# myfuns
import cuqi

%load_ext autoreload
%autoreload 2

# %%
TP = cuqi.testproblem.Deblur()
model = TP.model #Deblur model
data = TP.data #Data from deblur problem
sigma2 = TP.likelihood.std**2
n = model.domain_geometry.dim
m = model.range_geometry.dim

# %%
likelihood = cuqi.distribution.GaussianGen(model, sigma2)
prior      = cuqi.distribution.GaussianGen(0, 0.1*np.ones(n))

# %%
def potential(x):
    logpdf = -prior.logpdf(x)-likelihood(x=x).logpdf(data) 
    grad   = -prior.gradient(x)-likelihood.gradient(data,x=x)
    return logpdf,grad

# %%
solver = cuqi.solver.L_BFGS_B(potential,np.random.randn(n))
x_MAP = solver.solve()

# %%
model.domain_geometry.plot(x_MAP)
model.domain_geometry.plot(TP.exactSolution)

# %%
prior2 = cuqi.distribution.Gaussian(np.zeros(n), np.sqrt(0.1), np.eye(n))
TP.prior = prior2
x_MAP2 = TP.MAP()
model.domain_geometry.plot(x_MAP)
model.domain_geometry.plot(x_MAP2,'--')

# %%
