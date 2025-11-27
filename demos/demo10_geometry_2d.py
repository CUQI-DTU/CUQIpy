# %%
import sys
sys.path.append("..")
import cuqi
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

# %%
geom = cuqi.geometry.Continuous2D((100,40))
X, Y = np.meshgrid(geom.grid[0], geom.grid[1])
z = X**2 + Y**2
# %%
plt.figure(1)
geom.plot(np.ravel(z))

plt.figure(2)
geom.plot_contour(z)

plt.figure(3)
im = geom.plot_contourf(z)
plt.colorbar(im[0])

plt.figure(4)
ims = geom.plot(np.stack((np.ravel(z),np.ravel(-z)), axis=-1))

f1 = plt.figure(5)
ims = geom.plot_contour(np.stack((z,z,z), axis=-1))
for i, im in enumerate(ims):
    cax = make_axes_locatable(f1.axes[i]).append_axes('right', size='5%', pad=0.05)
    f1.colorbar(im, cax = cax)
