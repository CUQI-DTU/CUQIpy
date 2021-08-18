# %%
import sys
sys.path.append("..")
import cuqi
import matplotlib.pyplot as plt
import numpy as np

# %%
geom = cuqi.geometry.Continuous2D((100,40))
z = geom.grid[0]**2 + geom.grid[1]**2
# %%
plt.figure(1)
geom.plot(np.ravel(z))

plt.figure(2)
geom.plot_contour(z)

plt.figure(3)
im = geom.plot_contourf(z)
plt.colorbar(im)
