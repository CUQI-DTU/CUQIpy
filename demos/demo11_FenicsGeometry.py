#%%
import sys
sys.path.append("..")
import cuqi
import dolfin as dl
import matplotlib.pyplot as plt

#%% 1D case test
# Define FenicsContinuous2D object 
mesh = dl.UnitIntervalMesh(30)
Vh = dl.FunctionSpace(mesh, 'CG', 1)
geom = cuqi.fenicsGeometry.FenicsContinuous1D(Vh)

# Define function
f1= dl.Function(Vh)
f1.interpolate(dl.Expression("pow(sin(x[0]),3)", degree =2))

# plot
plt.figure(1)
geom.plot(f1.vector().get_local())

#%% 2D case test
# Define FenicsContinuous2D object 
mesh = dl.UnitSquareMesh(10,10)
Vh = dl.FunctionSpace(mesh, 'CG', 1)
geom = cuqi.fenicsGeometry.FenicsContinuous2D(Vh)

# Define function
f1= dl.Function(Vh)
f1.interpolate(dl.Expression("pow(sin(x[0]),2)*cos(x[1]*2)", degree =2))

# plot
plt.figure(2)
geom.plot(f1.vector().get_local())