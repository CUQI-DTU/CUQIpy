#%%
import sys

sys.path.append("..") 
import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt
import inspect

# Set rng seed 
np.random.seed(0)

#%%
import cuqi
from cuqi.testproblem import Deconvolution
from cuqi.model import LinearModel
from cuqi.distribution import Gaussian, Laplace_diff, Cauchy_diff
from cuqi.sampler import CWMH
from cuqi.problem import BayesianProblem
from cuqi.samples import Samples, Data, CUQIarray


#%%
dim = 128
kernel = ["Gauss","Sinc","vonMises"]
phantom = ["Gauss","Sinc","vonMises","Square","Hat","Bumps","DerivGauss"]
noise_type = ["Gaussian","ScaledGaussian"]
noise_std = 0.05

# Test problem
prob = Deconvolution(
    dim = dim,
    kernel=kernel[0],
    phantom=phantom[3],
    noise_type=noise_type[0],
    noise_std = noise_std
)

#%%
A = prob.model.get_matrix()
model = LinearModel(A)

# Define as linear model
#phantom = prob.exactSolution
#phantomD = Data(parameters=phantom,  geometry=model.domain_geometry)
phantomC = CUQIarray(prob.exactSolution,  geometry=model.domain_geometry)

phantomC

# %%
phantomC.plot(linestyle='--')

# %%
#data_cleanD = model(phantomD)
#data_cleanD.plot()

# %%
data_cleanC = model(phantomC)
data_cleanC.plot()


#%%
noise_std = 0.05
likelihood = Gaussian(model, noise_std, np.eye(dim))

likelihood(x=np.zeros(dim)).sample(5).plot()
plt.title('Noise samples'); plt.show()

#%%
data = likelihood(x=phantomC).sample()

#%%
prior_std = 0.2
prior = Gaussian(np.zeros(dim), prior_std, np.eye(dim))

# Plot samples of prior
prior.sample(5).plot()
plt.title('Realizations from prior'); plt.show()

#%%
IP = BayesianProblem(likelihood, prior, data)

#%%
x_MAP = IP.MAP() 

# Plot
#plt.plot(phantom,'.-')
#plt.plot(x_MAP,'.-')
phantomC.plot()
x_MAP.plot()
plt.title("MAP estimate")
plt.legend(["Exact","MAP"])
plt.show()

#%%
Ns = 5000   # Number of samples
result = IP.sample_posterior(Ns)

type(result)

#%%
result.plot_ci(95, exact=phantomC)

#%%
result.plot_std()

#%%
idx = [20,55,60]
result.plot_chain(idx)
plt.legend(idx)



# %% Now try heat

# domain definition
N = 128           # spatial discretization
L = 1
T = 0.2
skip = 1
KL_map = lambda x: x

model = cuqi.testproblem.Heat_1D(dim=N, endpoint=L, max_time=T, field_type=None, KL_map=KL_map).model
#model = cuqi.model.Heat_1D(N=N, L=L, T=T, field_type='KL', skip=skip)
x = model.domain_geometry.grid
x_data = x[::skip]
M = x_data.shape[0]

# constructing signal
true_init = 100*x*np.exp(-5*x)*np.sin(L-x)

# Signal as cuqi Data object
#true_initD = cuqi.samples.Data(funvals=true_init, geometry=model.domain_geometry)
true_initC = cuqi.samples.CUQIarray(true_init, is_par=False, geometry=model.domain_geometry)

# defining the heat equation as the forward map
#y_exact = model._advance_time(true_init) # observation vector
#y_exactC = model._advance_time(true_initC) # observation vector

y_exactC = model.forward(true_initC)

# %%
true_initC.plot()

#%%
y_exactC.plot()
# %%

model_step = cuqi.testproblem.Heat_1D(dim=N, endpoint=L, max_time=T, field_type='Step', KL_map=KL_map).model
#model_step = cuqi.model.Heat_1D(N=N, L=L, T=T, field_type='Step', skip=skip)

# %%
model_step.domain_geometry

# %%
true_stepC = CUQIarray(np.array([3,1,2]), geometry=model_step.domain_geometry)

# %%
true_stepC.plot()

# %%
y_stepC = model_step(true_stepC)

# %%
y_stepC.plot()


# %%  Heat with KL

model_KL = cuqi.testproblem.Heat_1D(dim=N, endpoint=L, max_time=T, field_type='KL', KL_map=KL_map).model

#%%
true_initKL = cuqi.samples.CUQIarray(true_init, is_par=False,  geometry=model_KL.domain_geometry)

# %% This should give an error as parameters are not available
true_initKL.parameters
