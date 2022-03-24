#%%

import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("../../")
import cuqi
import dolfin as dl

mesh = dl.UnitSquareMesh(20,20)
V = dl.FunctionSpace(mesh, 'CG', 1)
geometry = cuqi.fenics.geometry.FEniCSContinuous(V)
matern = cuqi.fenics.field.Matern(geometry, l = .2, nu = 2, num_terms=128)

dl.plot(matern.par2fun(np.random.rand(128)))


# Matern 2

#geometry = cuqi.geometry.Continuous2D(grid=(10,10))
#matern2 = cuqi.fenics.field.Matern(geometry, l = .2, nu = 2, num_terms=128)

#dl.plot(matern2.par2fun(np.random.rand(128)))

# %%
