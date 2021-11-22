# %% Initialize and import CUQI
sys.path.append("..") 
import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt
import cuqi

# Set rng seed 
np.random.seed(0)
# %% Import a deconvolution problem from file

# Load file with forward matrix, data and phantom
ref = np.load("data/Deconvolution.npz")

# Extract parameters
A = ref["A"]                # Matrix (forward model)
data = ref["data"]          # Data (noisy)
phantom = ref["phantom"]    # Phantom / ground truth
m,n = A.shape               # Dimensions of problem

# This deconvolution inverse problem looks like:
# b = A*x+e
# where e ~ Gaussian white noise.
# Suppose we know the standard deviation of noise is 0.05.

# %% Illustrate matrix (forward model)
plt.imshow(A); plt.title("Matrix"); plt.colorbar(); plt.show()

# %% Illustrate phantom + data
plt.plot(phantom); plt.title("Phantom (sinc function)"); plt.show()
plt.plot(data); plt.title("Measured (noisy) data"); plt.show()

# %% Set up CUQI problem forward model

# Define as linear model
model = cuqi.model.LinearModel(A)



# %% To carry out UQ we need to assume distributions
# for our random variables

# Define likelihood as Gaussian with model as mean and i.i.d. noise with 0.05 std.
noise_std = 0.05
likelihood = cuqi.distribution.Gaussian(model,noise_std,np.eye(m))

# Plot samples of noise
likelihood(x=np.zeros(n)).sample(5).plot()

plt.title('Noise samples'); plt.show()

# Plot samples of simulated data
likelihood(x=phantom).sample(5).plot()

plt.title('Simulated data'); plt.show()

# %% Define prior
prior_std = 0.2
prior = cuqi.distribution.Gaussian(np.zeros(n),prior_std,np.eye(n))

# Plot samples of prior
prior.sample(5).plot()
plt.title('Realizations from prior'); plt.show()


# %% Define cuqi (inverse) problem

# Bayesian model
IP = cuqi.problem.BayesianProblem(likelihood,prior,data)

# %% Compute MAP estimate.

x_MAP = IP.MAP() 

# Plot
plt.plot(phantom,'.-')
plt.plot(x_MAP,'.-')
plt.title("Map estimate")
plt.legend(["Exact","MAP"])
plt.show()


# %% Sample cuqi (inverse) problem
# Number of samples
Ns = 5000

# Sample posterior related to inverse problem

result = IP.sample_posterior(Ns)


# %% plot mean + 95% of samples

result.plot_ci(95,exact=phantom)


# %% Diagnostics

result.diagnostics()


# %% What happends if we change prior?

# Define correlation matrix where 30 closest neighbours are correlated
l = 30
corr = np.linspace(0,1,int(l/2)+1)
corr = np.hstack((corr,np.flipud(corr[:-1])))
indexes = np.linspace(-l/2,l/2,l+1,dtype=int)
corrmat = diags(corr, indexes, shape=(n, n)).toarray()

# Set new prior
IP.prior = cuqi.distribution.Gaussian(np.zeros(n),prior_std,corrmat)

# Plot samples from prior
IP.prior.sample(5).plot() 
plt.title('Realizations from prior'); plt.show()

# %% Sample new IP
result = IP.sample_posterior(Ns)

# plot mean + 95% of samples
result.plot_ci(95,exact=phantom)

# %% We provide test problems / prototype problems

tp = cuqi.testproblem.Deconvolution() #Default values


# %% Set up Deconvolution test problem
# Parameters for Deconvolution problem
dim = 128
kernel = ["Gauss","Sinc","vonMises"]
phantom = ["Gauss","Sinc","vonMises","Square","Hat","Bumps","DerivGauss"]
noise_type = ["Gaussian","ScaledGaussian"]
noise_std = 0.05

# Test problem
tp = cuqi.testproblem.Deconvolution(
    dim = dim,
    kernel=kernel[0],
    phantom=phantom[3],
    noise_type=noise_type[0],
    noise_std = noise_std
)


#%% Plot exact solution
plt.plot(tp.exactSolution,'.-'); plt.title("Exact solution"); plt.show()

# Plot data
plt.plot(tp.data,'.-'); plt.title("Noisy data"); plt.show()


# %% Lets try with the Gaussian prior

tp.prior = cuqi.distribution.Gaussian(np.zeros(n),prior_std,corrmat)

# Sample
result = tp.sample_posterior(Ns)

# plot mean + 95% of samples
result.plot_ci(95,exact=tp.exactSolution)

# %% Lets try with Cauchy prior

# Cauchy prior
scale = 2/n
tp.prior = cuqi.distribution.Cauchy_diff(np.zeros(n),scale,'neumann')

# Sample
result = tp.sample_posterior(Ns)

# plot mean + 95% of samples
result.plot_ci(95,exact=tp.exactSolution)


# %% Lets use some of the samplers directly from the sampler module

# Set up Laplace prior
loc = np.zeros(dim)
delta = 0.5
scale = delta*1/dim
prior = cuqi.distribution.Laplace_diff(loc,scale,'zero')

# Target and proposal
def target(x): return tp.likelihood(x=x).logpdf(tp.data)+prior.logpdf(x)
def proposal(x,scale): return np.random.normal(x,scale)

# Parameters for sampler
scale = 0.05*np.ones(dim)
x0 = 0.5*np.ones(dim)

# Define sampler (Component-Wise Metroplis-Hastings)
MCMC = cuqi.sampler.CWMH(target, proposal, scale, x0)

# Burn-in
Nb = int(0.2*Ns)   

# Run sampler (with adaptive parameter selection)
result = MCMC.sample_adapt(Ns,Nb)

# plot mean + 95 ci of samples using specific sampler
result.plot_ci(95,exact=tp.exactSolution)
