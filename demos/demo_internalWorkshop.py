# %% Initialize and import CUQI
sys.path.append("..") 
import numpy as np
import cuqi

# %%
# Import data + forward matrix
data  = np.load("data/Deconvolution.npz")["data"]  #Vector
A     = np.load("data/Deconvolution.npz")["A"]     #Matrix
m,n   = A.shape                                    #Dimension

# CUQI UQ in 5 lines of code.. 
model = cuqi.model.LinearModel(A)
noise = cuqi.distribution.Gaussian(np.zeros(m),0.05)
prior = cuqi.distribution.Gaussian(np.zeros(n),0.1)
IP = cuqi.problem.Type1(data,model,noise,prior) #data=model(prior)+noise
IP.UQ()

# %%
# CUQI testproblems (includes exact solution etc.)
TP = cuqi.testproblem.Deconvolution(phantom="sinc",prior=prior)
TP.UQ()

