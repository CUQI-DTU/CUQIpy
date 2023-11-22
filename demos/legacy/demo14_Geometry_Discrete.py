# %%
import sys

from numpy import random
sys.path.append("..")
import cuqi
import matplotlib.pyplot as plt
import numpy as np

# %% demo cuqi.geometry.Discrete 
geom = cuqi.geometry.Discrete(['a','b','c','e'])

plt.figure()
geom.plot([1,2,2,5])

plt.figure()
geom.plot([1,2,2,5],linestyle = '-')

plt.figure()
geom.plot([1,2,2,5], marker = '*')

# %% create uniform distribution 
x = cuqi.distribution.Uniform([2,1,2,0],[6,3,10,12])
xs = x.sample(20)


# %% plotting uniform distribution samples, chains and credibility interval using default geometry
#%% plot mean
xs.plot_mean()

# %% plot samples
xs.plot()

# %% plot samples
xs.plot(range(20))

# %% plot chain
xs.plot_chain(1)

# %% plot credibility interval
xs.plot_ci(95)
plt.figure()
xs.plot_ci(95, exact = [4,2,6,6])

# %% plotting uniform distribution samples, chains and credibility interval using the cuqi.geometry.Discrete geometry
xs.geometry = geom


#%% plot mean
xs.plot_mean()

# %% plot samples
xs.plot()

# %% plot samples
xs.plot(range(20))

# %% plot chain
xs.plot_chain(1)

# %% plot credibility interval
xs.plot_ci(95)
plt.figure()
xs.plot_ci(95, exact = [4,2,6,6])
