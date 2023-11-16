
#%%
import sys
sys.path.append("..") 
import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt
import cuqi


#%% Deconvolution Test problem
model_deconv, data_deconv, problem_info_deconv = cuqi.testproblem.Deconvolution1D().get_components()

#%% Deblur Test problem 
model_deblur, data_deblur, problem_info_deblur = cuqi.testproblem._Deblur().get_components()

#%%  Poisson1D Test problem
KL_map = lambda x: np.exp(x)
f = lambda xs: 10*np.exp( -( (xs - 0.5)**2 ) / 0.02) 

model_poisson, data_poisson, problem_info_poisson = cuqi.testproblem.Poisson1D(dim=501, endpoint=1, source=f, field_type=None, map=KL_map).get_components()
