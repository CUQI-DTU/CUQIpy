# %% Initialize and import CUQI
import sys
sys.path.append("..") 
import numpy as np
import cuqi
from cuqi.model import LinearModel
from cuqi.distribution import Gaussian, Cauchy_diff
from cuqi.problem import BayesianProblem

# %% Minimal example

# Import data and forward matrix
A      = np.load("data/Deconvolution.npz")["A"]          #Matrix (numpy)
b      = np.load("data/Deconvolution.npz")["data"]       #Vector (numpy)
m,n    = A.shape

# Data from square function
b = cuqi.testproblem.Deconvolution1D(phantom="square").data
x_exact = cuqi.testproblem.Deconvolution1D(phantom="square").exactSolution

# Set up Bayesian model for inverse problem
model      = LinearModel(A)                           #Model for inverse problem
prior      = Gaussian(np.zeros(n),0.1)                #Prior distribution
likelihood = Gaussian(model,0.05**2).to_likelihood(b)    #Likelihood function
IP         = BayesianProblem(likelihood, prior)        #Bayesian model for inverse problem
IP.UQ(exact=x_exact)                                  #Perform UQ on inverse problem

# %%
# Wrap into CUQI "testproblem".
TP = cuqi.testproblem.Deconvolution1D(prior=prior)
TP.UQ()

# %%
# switch prior
# Set up Bayesian model for inverse problem
M  = LinearModel(forward=A)                                          #Model for inverse problem
P  = Cauchy_diff(location=np.zeros(n),scale=0.05,bc_type='neumann')  #Prior distribution
L  = Gaussian(mean=M,cov=0.05**2).to_likelihood(data=b)             #Likelihood function
IP = BayesianProblem(likelihood=L, prior=P)                          #Bayesian model for inverse problem
IP.UQ(exact=x_exact)                                                 #Perform UQ on inverse problem

# %%
samples = IP.sample_posterior(50000)
# %%
samples.plot_ci(95,exact=x_exact)
# %%
# Set up Bayesian model for inverse problem
model      = LinearModel(A)                             #Model for inverse problem
prior      = Cauchy_diff(np.zeros(n),0.05,'neumann')    #Prior distribution
likelihood = Gaussian(model,0.05**2).to_likelihood(data=b) #Likelihood function
IP         = BayesianProblem(likelihood,prior)          #Bayesian model for inverse problem
IP.UQ()                                                 #Perform UQ on inverse problem        
