#%%
import sys
sys.path.append("..")
import cuqi
import dolfin as dl
import matplotlib.pyplot as plt
import numpy as np

#%% 1D case test
# Define FenicsContinuous2D object 
mesh = dl.UnitIntervalMesh(30)
Vh = dl.FunctionSpace(mesh, 'CG', 1)
geom = cuqi.fenicsGeometry.FenicsContinuous(Vh)

# Define function
f1= dl.Function(Vh)
f1.interpolate(dl.Expression("pow(sin(x[0]),3)", degree =2))

# plot
plt.figure(1)
geom.plot(f1.vector().get_local())

plt.figure(2)
geom.plot(np.stack((f1.vector().get_local(), f1.vector().get_local()), axis = -1), subplots = False)

#%% 2D case test
# Define FenicsContinuous2D object 
mesh = dl.UnitSquareMesh(10,10)
Vh = dl.FunctionSpace(mesh, 'CG', 1)
geom = cuqi.fenicsGeometry.FenicsContinuous(Vh)

# Define function
f1= dl.Function(Vh)
f1.interpolate(dl.Expression("pow(sin(x[0]),2)*cos(x[1]*2)", degree =2))

# plot
plt.figure(3)
geom.plot(f1.vector().get_local())

plt.figure(4)
geom.plot(np.stack((f1.vector().get_local(), f1.vector().get_local()), axis = -1))

