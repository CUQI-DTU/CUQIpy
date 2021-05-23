# %% Initialize and import CUQI
import sys
sys.path.append("..") 
import numpy as np
import cuqi

# %%
x = cuqi.distribution.Normal2("x",0,1)
x.sample(1)

# %%
x = cuqi.distribution.Normal2("x",0,1)
x.sample(2)

#%%
y = cuqi.distribution.Normal2("y",0,x)
y.sample(1,cond={"x":2})

# %%
y.sample(1)
# %%
