# %% Here we test the automatic sampler selection for Deconvolution (1D and 2D).
import sys
sys.path.append("..")
import cuqi
import numpy as np

# %% Deconvolution 1D

TP = cuqi.testproblem.Deconvolution()
TP.prior = cuqi.distribution.Gaussian(mean=np.zeros(TP.model.domain_dim),
                                      std=1,
                                      geometry=TP.model.domain_geometry)
