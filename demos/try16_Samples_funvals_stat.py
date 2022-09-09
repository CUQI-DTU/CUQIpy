# Demonstrate computation of Samples statistics on both function values and parameter values.
# %%
import sys
sys.path.append("..")
import cuqi
import numpy as np
import matplotlib.pyplot as plt


samples = cuqi.samples.Samples(3*np.random.rand(10, 15), geometry = cuqi.geometry.MappedGeometry(cuqi.geometry.Continuous1D(10), map=lambda x:x**2))

# Compute statistics on parameter values
samples.plot_ci(95, plot_par = True) #plot in parameter space

plt.figure()
samples.plot_ci(95) #plot in function space


# Compute statistics on function values
func_samples = samples.funvals
plt.figure()
func_samples.plot_ci(95) #plot in function space


plt.figure()
func_samples.plot_ci(95, plot_par=True) #Raises and error: plot in parameter space
