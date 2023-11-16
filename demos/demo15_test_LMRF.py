# %%
import sys
sys.path.append("..")
import cuqi
import numpy as np
import matplotlib.pyplot as plt
import scipy
import cuqi.operator
# %%
location = 0
N = 2
scale = 1.0
dom = 1
BCs = 'neumann'
x = cuqi.distribution.LMRF(location, scale, BCs, geometry=N)
print(x.logd(np.array([3,4])))
# %%
location = 0
scale = 1
y = cuqi.distribution.Laplace(location, scale)
print(y.pdf(6))
print(y.pdf(1))
samples = y.sample(1000)

# %%
plt.figure()
samples.plot_chain(0)
#samples.plot()
plt.figure()
#plt.plot(samples.samples,[y.pdf(s) for s in samples.samples], 'o')

plt.hist(samples.samples.T,bins = 30)
# %%

