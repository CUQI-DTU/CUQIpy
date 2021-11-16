
#%%
sys.path.append("..") 
import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt
import cuqi

# %% Set up Deconvolution test problem
# Parameters for Deconvolution problem
dim = 128
kernel = ["Gauss","Sinc","vonMises"]
phantom = ["Gauss","Sinc","vonMises","Square","Hat","Bumps","DerivGauss"]
noise_type = ["Gaussian","ScaledGaussian"]
noise_std = 0.05

#%% Deconvolution Test problem
model, data, problem_info = cuqi.testproblem.Deconvolution.get_components(
    dim = dim,
    kernel=kernel[0],
    phantom=phantom[3],
    noise_type=noise_type[0],
    noise_std = noise_std
)

#%% Deblur Test problem 
model, data, problem_info = cuqi.testproblem.Deblur.get_components()

#%%  Poisson_1D Test problem
KL_map = lambda x: np.exp(x)
f = lambda xs: 10*np.exp( -( (xs - 0.5)**2 ) / 0.02) 

model, data, problem_info = cuqi.testproblem.Poisson_1D.get_components(dim=501, endpoint=1, source=f, field_type=None, KL_map=KL_map)
