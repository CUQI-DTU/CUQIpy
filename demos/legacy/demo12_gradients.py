# %%
import sys
sys.path.append("../")
import numpy as np
import matplotlib.pyplot as plt
import cuqi

# %%
# Load cuqi deblur model and data
TP = cuqi.testproblem._Deblur()
n = TP.model.domain_dim
m = TP.model.range_dim
x_true = TP.exactSolution

# %%
# Define priors
pr = 'gaussian'
if (pr == 'gaussian'):
    var = 0.2
    prior = cuqi.distribution.Gaussian(np.zeros(n), var)
elif (pr == 'cauchy'):
    h = 1/n
    delta = 0.3
    prior = cuqi.distribution.CMRF(np.zeros(n), delta*h, 'neumann')

# %%
# Compare with MAP computed using BayesianProblem (within the testproblem)
x0 = np.random.randn(n)
TP.prior = prior
if (pr == 'gaussian'):
    x_MAP = TP.MAP()
else:
    # Solve posterior problem using BFGS
    def f(x): return -TP.posterior.logd(x)
    def gradf(x): return -TP.posterior.gradient(x)
    solver = cuqi.solver.L_BFGS_B(f, x0, gradf)
    x_MAP, solution_info = solver.solve()
print('relative error MAP:', np.linalg.norm(x_MAP-x_true)/np.linalg.norm(x_true))

# %%
# sampling using NUTS
MCMC = cuqi.sampler.NUTS(TP.posterior)
Ns = int(200)      # number of samples
Nb = int(0.2*Ns)   # burn-in
samples = MCMC.sample(Ns, Nb)
#
xs = samples.samples
x_mean = np.mean(xs, axis=1)
x_std = np.std(xs, axis=1)
print('relative error mean:', np.linalg.norm(x_mean-x_true)/np.linalg.norm(x_true))

# %%
# plots
samples.plot_ci(exact=x_true)
TP.model.domain_geometry.plot(x_MAP, 'b-', label='MAP')
plt.legend(['Mean', 'True', 'MAP', 'CI'])
plt.show()

# %%
