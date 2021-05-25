# %% Initialize and import CUQI
import sys
sys.path.append("..") 
import numpy as np
import cuqi

# %% Minimal example

# Import data and forward matrix
A      = np.load("data/Deconvolution.npz")["A"]          #Matrix (numpy)
b      = np.load("data/Deconvolution.npz")["data"]       #Vector (numpy)
m,n    = A.shape

# Set up Bayesian model for inverse problem
model      = cuqi.model.LinearModel(A)                                  #Model for inverse problem
prior      = cuqi.distribution.Gaussian(np.zeros(n),0.1)                #Prior distribution
likelihood = cuqi.distribution.Gaussian(model,0.05,np.eye(m))           #Likelihood distribution
IP         = cuqi.problem.BayesianModel(likelihood,prior,data=b)  #Bayesian model for inverse problem
IP.UQ()                                                                 #Perform UQ on inverse problem

# %%
# Wrap into CUQI "testproblem".
TP = cuqi.testproblem.Deconvolution(phantom="sinc",prior=prior)
TP.UQ()
