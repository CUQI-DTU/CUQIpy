# %% Initialize and import CUQI
sys.path.append("..") 
import numpy as np
import cuqi

# This makes sure modules are reloaded
# each time a function is called (optional)
%load_ext autoreload
%autoreload 2

# %%
# Import data + forward matrix
data  = np.load("data/Deconvolution.npz")["data"]    #Vector
A     = np.load("data/Deconvolution.npz")["A"]       #Matrix
xtrue = np.load("data/Deconvolution.npz")["phantom"] #Exact solution
m,n   = A.shape                                      #Dimension

# CUQI UQ in a few lines of code.. 
model = cuqi.model.LinearModel(A)
noise = cuqi.distribution.Gaussian(np.zeros(m),0.05)
prior = cuqi.distribution.Gaussian(np.zeros(n),0.1)
IP = cuqi.problem.Type1(data,model,noise,prior) #data=model(prior)+noise
IP.UQ(exact=xtrue)

# %% Minimal example

# Import data + forward matrix
A     = np.load("data/Deconvolution.npz")["A"]     #Matrix
b_obs  = np.load("data/Deconvolution.npz")["data"]    #Vector
m,n   = A.shape

# Set up abstract problem (both additive and multiplicative noise)
prior = cuqi.distribution.Gaussian(np.zeros(n),0.1) #Prior
model = cuqi.model.LinearModel(A) #Forward model
data = cuqi.distribution.Gaussian(lambda A,x: A*x,0.05,np.eye(m)) #data/likelihood
IP = cuqi.problem.BayesianModel(data,model,prior) #data=model(prior)
#IP.UQ()..

# Solve inverse problem and plot
samples = IP.sample_posterior(2500,data=b_obs)
samples.plot_ci(95)

# %% Generate data and solve specific example

# Read ground truth
phantom = np.load("data/Deconvolution.npz")["phantom"]

# Generate some data according to distribution of b
b_obs = b.sample(1,A=model,x=phantom)

# MAP estimate
x_MAP = IP.MAP(data=b_obs)

# Sample posterior
samples = IP.sample_posterior(2500,data=b_obs)

# Plot confidense interval of samples
samples.plot_ci(95,exact=phantom)


# %%
# Wrap into CUQI "testproblem".
TP = cuqi.testproblem.Deconvolution(phantom="sinc",prior=prior)
TP.UQ()


# %%
