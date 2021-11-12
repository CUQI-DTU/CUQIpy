# %% Initialize and import CUQI
import sys
sys.path.append("..") 
import numpy as np
import cuqi
import matplotlib.pyplot as plt

# %%
class GaussianGrad(cuqi.distribution.Gaussian):
    def grad(self,x,data):
        model = self.mean #HACK FOR NOW TODO make model accesible in dist???
        misfit = data-model.forward(x)
        lambda_obs = 1/(self.std**2)
        return lambda_obs*model.adjoint(misfit)
    
class CauchyGrad(cuqi.distribution.Cauchy_diff):
    def grad(self,x):
        diff = self.D @ x
        c = self.scale
        return (-2*diff/(diff**2+c**2)) @ self.D



# %%
test = cuqi.testproblem.Deblur()
n = test.model.domain_dim
h = test.meshsize

# Extract data
data = test.data

#Likelihood
mean = test.likelihood.mean
std = .1#test.likelihood.std
corrmat = test.likelihood.Sigma
likelihood = GaussianGrad(mean,std,corrmat)

# Prior
loc = np.zeros(n)
delta = 1
scale = delta*h
prior = CauchyGrad(loc, scale, 'neumann')

# %%
x0 = cuqi.distribution.Gaussian(np.zeros(n),1).sample()
posterior = cuqi.distribution.Posterior(likelihood,prior,data)
MCMC = cuqi.sampler.NUTS(posterior,x0)

# %%
samples = MCMC.sample(10,10)

# %%
xs = samples[0]

# %%
x_mean = np.mean(xs.samples,axis=1)
# %%
plt.plot(x_mean)
plt.plot(test.exactSolution)
#%%
results = xs
# %%
results.plot_ci(95,exact=test.exactSolution)
# %%
