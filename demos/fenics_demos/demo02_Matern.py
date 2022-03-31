#%%
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("../../")
from cuqi.fenics.geometry import MaternExpansion, FEniCSContinuous
from cuqi.distribution import GaussianCov
import dolfin as dl

mesh = dl.UnitSquareMesh(20,20)
V = dl.FunctionSpace(mesh, 'CG', 1)
geometry = FEniCSContinuous(V)
MaternGeometry = MaternExpansion(geometry, 
                                length_scale = .2,
                                num_terms=128)

MaternField = GaussianCov(np.zeros(MaternGeometry.dim),
                cov=np.eye(MaternGeometry.dim),
                geometry= MaternGeometry)

samples = MaternField.sample()
samples.plot()

# View the first 10 eigenvectors
for i in range(10):
    plt.figure()
    geometry.plot(MaternGeometry.eig_vec[:,i]) 
    plt.show()
    plt.close('all')