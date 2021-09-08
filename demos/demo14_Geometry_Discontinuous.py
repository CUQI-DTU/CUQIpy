# %%
import sys
sys.path.append("..")
import cuqi
import matplotlib.pyplot as plt
import numpy as np

# %%
geom = cuqi.geometry.Discontinuous(4, labels=['a','b','c','e'])
plt.figure()
geom.plot([1,2,2,5],linestyle = '-')

plt.figure()
geom.plot([1,2,2,5], marker = '*')

plt.figure()
geom.plot([1,2,2,5])

plt.figure()
geom.plot([1,2,2,5])

# %%
x = cuqi.distribution.Uniform([2,1,2,0],[6,3,10,12])
xs = x.sample(20)
xs.geometry = geom
#%%

xs.plot_mean() # Should compute mean and plot according to the geometry of the object
#xs.geometry.labels = None
plt.figure()
xs.plot_mean()

# %%
xs.plot()

# %%
xs.plot_chain(1)

# %%
xs.plot_ci(95)
plt.figure()
xs.plot_ci(95, exact = [4,2,6,8])
