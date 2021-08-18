# %% Initialize and import CUQI
sys.path.append("..") 
import numpy as np
import cuqi

# This makes sure modules are reloaded
# each time a function is called (optional)
%load_ext autoreload
%autoreload 2


# %% Minimal example

# Import data + forward matrix
A     = np.load("data/Deconvolution.npz")["A"]     #Matrix
m,n   = A.shape     

# Set up abstract model
x = cuqi.distribution.Gaussian(np.zeros(n),0.1)                # Prior
A = cuqi.model.LinearModel(A)                                  # Model
b = cuqi.distribution.Gaussian(lambda A,x: A*x,0.05,np.eye(m)) # Data
IP = cuqi.problem.BayesianModel(b,A,x)                         # Bayesian Model

# Generate some data
#b_obs = b.sample(1,A=A,x=np.ones(n))
b_obs  = np.load("data/Deconvolution.npz")["data"]  #Vector

# MAP
x_MAP = IP.MAP(data=b_obs)

# Sample inverse problem with given data
s = IP.sample_posterior(2500,data=b_obs)

# Plot confidense interval
s.plot_ci(95)


# %%
# Import data + forward matrix
data  = np.load("data/Deconvolution.npz")["data"]  #Vector
A     = np.load("data/Deconvolution.npz")["A"]     #Matrix
m,n   = A.shape                                    #Dimension

# CUQI UQ in a few lines of code.. 
model = cuqi.model.LinearModel(A)
noise = cuqi.distribution.Gaussian(np.zeros(m),0.05)
prior = cuqi.distribution.Gaussian(np.zeros(n),0.1)
IP = cuqi.problem.Type1(data,model,noise,prior) #data=model(prior)+noise
IP.UQ()

# %%
# Wrap into CUQI "testproblem".
TP = cuqi.testproblem.Deconvolution(phantom="sinc",prior=prior)
TP.UQ()


# %%
