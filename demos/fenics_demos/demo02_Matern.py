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
matern = cuqi.fenics.geometry.Matern(geometry, l = .2, num_terms=128)

dl.plot(matern.par2fun(np.random.rand(128)))
