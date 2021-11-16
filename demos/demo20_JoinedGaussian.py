# %%
import sys
sys.path.append("..") 
import cuqi
import numpy as np
import matplotlib.pyplot as plt

# %%
# Define two Gaussians by the mean and sqrtprec
n = 128

X1 = cuqi.distribution.GaussianCov(np.ones(n),np.eye(n))
X2 = cuqi.distribution.GMRF(3*np.ones(n),1,n,1,'zero')

# %%
# Define joined Gaussian

X = cuqi.distribution.JoinedGaussianSqrtPrec([X1.mean,X2.mean],[X1.sqrtprec,X2.sqrtprec])

# %%
# Use this Gaussian as prior in test problem
TP = cuqi.testproblem.Deconvolution(prior=X)
TP.UQ()

# %%
# Maximum a posteriori estimation (logpdf is not implemented, so throws an error.)
#x_MAP = TP.MAP()
