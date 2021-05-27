# %% Initialize and import CUQI
import sys
sys.path.append("..") 
import numpy as np
import cuqi

# %%
x = cuqi.distribution.Normal2(0,1)
x.sample(1)

#%%
y = cuqi.distribution.Normal2(0,x)
y(std=10).sample(1)

# %%
y.sample(1)

