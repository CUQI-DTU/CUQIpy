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
matern = cuqi.fenics.geometry.Matern(geometry, length_scale = .2, num_terms=128)

x = cuqi.samples.CUQIarray(np.random.randn(128), geometry=matern)
x.plot()