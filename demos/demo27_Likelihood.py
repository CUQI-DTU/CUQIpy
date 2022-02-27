# %% Import CUQI and other packages
import sys
sys.path.append("..")
import cuqi
import matplotlib.pyplot as plt
import numpy as np

# %% Define Gaussian distribution + data vector
TP = cuqi.testproblem.Deconvolution()
data_dist = TP.likelihood # Old style likelihood
data = TP.data
n = TP.model.domain_dim

# %%
model = TP.model #Model data etc.
#data_dist = cuqi.distribution.GaussianCov(model, lambda sigma: sigma]) # this is also the likelihood dist.
likelihood = cuqi.likelihood.Likelihood(data_dist, data)
likelihood

# %%
gt = TP.exactSolution
likelihood(x=gt); 
likelihood.gradient(x=gt); 
# %%
f = lambda x: data_dist(x=x).logpdf(data) #Custom function
g = lambda x: data_dist.gradient(data, x=x)
likelihoodU = cuqi.likelihood.UserDefinedLikelihood(dim=n, logpdf_func=f, gradient_func=g)
likelihoodU
# %%
likelihoodU(x=gt); 
likelihoodU.gradient(x=gt); 
# %% Five line example (No change yet)
model = TP.model #Model data etc.
prior = cuqi.distribution.Gaussian(mean=np.zeros(n), std=0.2)
data_dist = cuqi.distribution.Gaussian(model, 0.05)
IP = cuqi.problem.BayesianProblem(data_dist, prior, data)
IP.UQ(exact=TP.exactSolution)

# %% Five line example (Make to_likelihood method)
model = TP.model #Model data etc.
prior = cuqi.distribution.Gaussian(mean=np.zeros(n), std=0.2)
#likelihood = cuqi.distribution.Gaussian(model, 0.05).to_likelihood(data) # Avoid naming the conditional distribution :D
IP = cuqi.problem.BayesianProblem(likelihood, prior)
IP.UQ(exact=TP.exactSolution)

# %% Posterior (remains with distribution..)

# posterior = likelihood x prior / evidence

#posterior = cuqi.distribution.Posterior(data_dist, prior, data) # !!Kill your darlings!!
#posterior = cuqi.distribution.Posterior(likelihood, prior)

# %% Minimize.. #ToDo. Fix distribution allowing args
#x0 = np.zeros(n)
#solver = cuqi.solver.minimize(likelihood, x0, gradfunc=likelihood.gradient)
#x_ML = solver.solve() 

# %% PCN..


# %%
