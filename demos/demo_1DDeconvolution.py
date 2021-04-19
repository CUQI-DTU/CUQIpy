# %% Initialize and import CUQI
sys.path.append("..") 
import numpy as np
import matplotlib.pyplot as plt
import cuqi

# This makes sure modules are reloaded each time a function is called (optional)
%load_ext autoreload
%autoreload 2

# Set rng seed 
np.random.seed(0)
# %% Load file with forward matrix and data:
ref = np.load("data/Deconvolution.npz")
A = ref["A"]                # Matrix (forward model) - Convolution with Gaussian kernel
data = ref["data"]          # Data (noisy)
phantom = ref["phantom"]     # Phantom / ground truth
m,n = A.shape               # Dimensions of problem

# This deconvolution inverse problem looks like:
# b = A*x+e
# where e ~ Gaussian white noise.

# %% Illustrate matrix (forward model)
plt.imshow(A); plt.title("Matrix (convolution w. Gaussian kernel)"); plt.colorbar(); plt.show()

# %% Illustrate phantom + data
plt.plot(phantom); plt.title("Phantom (sinc function)"); plt.show()
plt.plot(data); plt.title("Measured (noisy) data"); plt.show()

# %% Set up CUQI problem forward model

# Define as linear model
model = cuqi.model.LinearModel(A)

# %% Define distributions

#  Define noise
noise_std = 0.05
noise = cuqi.distribution.Gaussian(np.zeros(m),noise_std,np.eye(m))

# Define prior
prior_std = 0.1
prior = cuqi.distribution.Gaussian(np.zeros(n),prior_std,np.eye(n))

# %% Define cuqi (inverse) problem

# Type1: data = model(x)+noise
IP = cuqi.problem.Type1(data,model,noise,prior)

# %% Compute MAP estimate.
# Depending on defined model, noise and prior a method for estimating map is selected.
x_MAP = IP.MAP() 

# Plot
plt.plot(phantom,'.-')
plt.plot(x_MAP,'.-')
plt.title("Map estimate")
plt.legend(["Exact","MAP"])
plt.show()


# %% Sample cuqi (inverse) problem
# Number of samples
Ns = 25000

# Sample (depending on defined model, noise and prior a sampler is selected)
result = IP.sample(Ns)

# plot mean + 95% of samples
x_mean = np.mean(result,axis=1)
x_lo95, x_up95 = np.percentile(result, [2.5, 97.5], axis=1)

plt.plot(phantom,'.-')
plt.plot(x_mean,'.-')
plt.fill_between(np.arange(n),x_up95, x_lo95, color='dodgerblue', alpha=0.25)
plt.title("Posterior samples")
plt.legend(["Exact","Posterior mean","Confidence interval"])
plt.show()

# %% What happends if we change prior?

# Set new prior
#IP.prior = cuqi.distribution.Cauchy_diff(np.zeros(n),10/n,'neumann')
IP.prior = cuqi.distribution.GMRF(np.zeros(n), 10, n, 1, 'zero')
# Sample
result = IP.sample(Ns)

# plot mean + 95% of samples
x_mean = np.mean(result,axis=1)
x_lo95, x_up95 = np.percentile(result, [2.5, 97.5], axis=1)

plt.plot(phantom,'.-')
plt.plot(x_mean,'.-')
plt.fill_between(np.arange(n),x_up95, x_lo95, color='dodgerblue', alpha=0.25)
plt.title("Posterior samples")
plt.legend(["Exact","Posterior mean","Confidence interval"])
plt.show()

# %%
# Look at chains etc.

# %% Set up Deconvolution test problem
# Parameters
dim = 128
kernel = ["Gauss","Sinc","vonMises"]
phantom = ["Gauss","Sinc","vonMises","Square","Hat","Bumps","DerivGauss"]
noise_type = ["Gaussian","xxxGaussian"]
noise_std = 0.1

# Test problem
tp = cuqi.testproblem.Deconvolution(
    dim = dim,
    kernel=kernel[0],
    phantom=phantom[1],
    noise_type=noise_type[0],
    noise_std = noise_std
)

#%% Plot exact solution
plt.plot(tp.exactSolution,'.-'); plt.title("Exact solution"); plt.show()

# Plot data
plt.plot(tp.data,'.-'); plt.title("Noisy data"); plt.show()

# %% Try sampling using a specific sampler
# Set up target and proposal
loc = np.zeros(dim)
delta = 4
scale = delta*1/dim
prior = cuqi.distribution.Cauchy_diff(loc, scale, 'neumann')
#prior = cuqi.distribution.Laplace_diff(loc,scale,'zero')
def target(x): return tp.likelihood.logpdf(tp.data,x)+prior.logpdf(x)
def proposal(x,scale): return np.random.normal(x,scale)

# Parameters
scale = 0.05*np.ones(dim)
x0 = 0.5*np.ones(dim)

# Define sampler
MCMC = cuqi.sampler.CWMH(target, proposal, scale, x0)

Nb = int(0.4*Ns)   # burn-in
# Run sampler
ti = time.time()
result2, target_eval, acc = MCMC.sample_adapt(Ns,Nb)
print('Elapsed time:', time.time() - ti)

# %% plot mean + 95 ci of samples using specific sampler
x_mean_2 = np.mean(result2,axis=1)
x_lo95_2, x_up95_2 = np.percentile(result2, [2.5, 97.5], axis=1)

plt.plot(tp.exactSolution)
plt.plot(x_mean_2)
plt.fill_between(np.arange(tp.model.dim[1]),x_up95_2, x_lo95_2, color='dodgerblue', alpha=0.25)
plt.title("Posterior samples using sampler")
plt.legend(["Exact","Posterior mean","Confidense interval"])


# %%
