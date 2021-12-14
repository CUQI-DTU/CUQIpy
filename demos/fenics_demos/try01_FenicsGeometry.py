
#%%
import sys
sys.path.append("../../")
import cuqi
import matplotlib.pyplot as plt
import numpy as np
import warnings

try: 
    import dolfin as dl
except Exception as error:
    warnings.warn(error.msg)

#%% 1D case test
# Define FenicsContinuous2D object 
mesh1 = dl.UnitIntervalMesh(30)
Vh1 = dl.FunctionSpace(mesh1, 'CG', 1)
geom1 = cuqi.fenics.geometry.FEniCSContinuous(Vh1)

# Define function
f1= dl.Function(Vh1)
f1.interpolate(dl.Expression("pow(sin(x[0]),3)", degree =2))

# plot
plt.figure(1)
geom1.plot(f1.vector().get_local())

plt.figure(2)
geom1.plot(np.stack((f1.vector().get_local(), f1.vector().get_local()), axis = -1), subplots = True)

#%% 2D case test
# Define FenicsContinuous2D object 
mesh = dl.UnitSquareMesh(10,10)
Vh = dl.FunctionSpace(mesh, 'CG', 1)
geom = cuqi.fenics.geometry.FEniCSContinuous(Vh)

# Define function
f1= dl.Function(Vh)
f1.interpolate(dl.Expression("pow(sin(x[0]),2)*cos(x[1]*2)", degree =2))

# plot
plt.figure(3)
geom.plot(f1.vector().get_local())

plt.figure(4)
geom.plot(np.stack((f1.vector().get_local(), f1.vector().get_local()), axis = -1))


# %%
