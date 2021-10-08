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
n = test.model.dim[1]
h = test.meshsize

# Extract data
data = test.data

#Likelihood
mean = test.likelihood.mean
std = test.likelihood.std
corrmat = test.likelihood.R
likelihood = GaussianGrad(mean,std,corrmat)

# Prior
loc = np.zeros(n)
delta = 1
scale = delta*h
prior = CauchyGrad(loc, scale, 'neumann')

# %%
x0 = cuqi.distribution.Gaussian(np.zeros(n),1).sample()
MCMC = cuqi.sampler.NUTS(likelihood,prior,data,x0)

# %%
samples = MCMC.sample(1000,500)

# %%
xs = samples[0]

# %%
x_mean = np.mean(xs,axis=1)
# %%
plt.plot(x_mean)
plt.plot(test.exactSolution)
#%%
results = cuqi.samples.Samples(xs)
# %%
results.plot_ci(95,exact=test.exactSolution)