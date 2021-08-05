# %%
import sys
sys.path.append("..")
import cuqi
import matplotlib.pyplot as plt
import numpy as np

# %%
geom = cuqi.geometry.Continuous2D((100,40))

# %%
plt.figure(1)
geom.plot(np.ravel(geom.grid[0]))

plt.figure(2)
geom.plot_contour(geom.grid[0])

plt.figure(3)
geom.plot_contourf(geom.grid[0])
