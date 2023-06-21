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
from cuqi.testproblem import Deconvolution1D
from cuqi.model import LinearModel
from cuqi.distribution import Gaussian, LMRF, CMRF
from cuqi.sampler import CWMH
from cuqi.problem import BayesianProblem
from cuqi.samples import Samples
from cuqi.array import CUQIarray


#%%
dim = 128
kernel = ["Gauss","Sinc","vonMises"]
phantom = ["Gauss","Sinc","vonMises","Square","Hat","Bumps","DerivGauss"]
noise_type = ["Gaussian","ScaledGaussian"]
noise_std = 0.05

# Test problem
prob = Deconvolution1D(
    dim = dim,
    PSF=kernel[0],
    phantom=phantom[3],
    noise_type=noise_type[0],
    noise_std = noise_std
)

#%%
A = prob.model.get_matrix()
model = LinearModel(A)

# Define as linear model
phantomC = CUQIarray(prob.exactSolution,  geometry=model.domain_geometry)
phantomC

# %%
phantomC.plot(linestyle='--')

# %%
data_cleanC = model(phantomC)
data_cleanC.plot()

#%% Check adjoint
z = model.adjoint(data_cleanC)
z

#%%
z.plot()

#%%
noise_std = 0.05
data_dist = Gaussian(model, noise_std, np.eye(dim))

data_dist(x=np.zeros(dim)).sample(5).plot()
plt.title('Noise samples'); plt.show()

#%%
data = data_dist(x=phantomC).sample()

#%%
prior_std = 0.2
prior = Gaussian(np.zeros(dim), prior_std, np.eye(dim), name="x")

# Plot samples of prior
prior.sample(5).plot()
plt.title('Realizations from prior'); plt.show()

#%%

likelihood = data_dist.to_likelihood(data)

IP = BayesianProblem(likelihood, prior)

#%%
x_MAP = IP.MAP() 

# Plot
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

model = cuqi.testproblem.Heat1D(dim=N, endpoint=L, max_time=T, field_type=None).model
x = model.domain_geometry.grid
x_data = x[::skip]
M = x_data.shape[0]

# constructing signal
true_init = 100*x*np.exp(-5*x)*np.sin(L-x)

# Signal as cuqi Data object
true_initC = cuqi.array.CUQIarray(true_init, is_par=False, geometry=model.domain_geometry)

# defining the heat equation as the forward map
y_exactC = model.forward(true_initC)

# %%
true_initC.plot()

# %%
true_initC.plot(plot_par=True)

#%%
y_exactC.plot()

#%%
y_exactC.plot(plot_par=True)

# %%  Now heat with step 
model_step = cuqi.testproblem.Heat1D(dim=N, endpoint=L, max_time=T, field_type='Step').model

# %%
model_step.domain_geometry

# %%
true_stepC = CUQIarray(np.array([3,1,2]), geometry=model_step.domain_geometry)

# %%
true_stepC.plot()

#%% 
true_stepC.plot(plot_par=True)

# %%
y_stepC = model_step(true_stepC)

# %%
y_stepC.plot()

# %%  Heat with KL
model_KL = cuqi.testproblem.Heat1D(dim=N, endpoint=L, max_time=T, field_type='KL').model

#%%
true_initKL = cuqi.array.CUQIarray(true_init, is_par=False,  geometry=model_KL.domain_geometry)

# %% This should give an error as parameters are not available
true_initKL.parameters

# %% Plotting parameters should also give an error
true_initKL.plot(plot_par=True)

# %% Plotting function values is available
true_initKL.plot()

# %% 
y_exactKL = model_KL(true_initKL)
y_exactKL.plot()

# %%  Range geometry is default so fun2par is available and parameter plotting works
y_exactKL.plot(plot_par=True)
