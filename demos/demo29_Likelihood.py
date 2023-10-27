# %% Import CUQI and other packages
import sys
sys.path.append("..")
import cuqi
import matplotlib.pyplot as plt
import numpy as np

from cuqi.testproblem import Deconvolution1D
from cuqi.distribution import Gaussian
from cuqi.problem import BayesianProblem

# Parameters
n = 128

# %% Five line example (likelihood + prior)
model, data, probInfo = Deconvolution1D(dim=n, phantom="Square").get_components()
prior = Gaussian(mean=np.zeros(n), sqrtcov=0.2, name="x")
likelihood = Gaussian(mean=model, sqrtcov=0.05, name="y").to_likelihood(data)
IP = BayesianProblem(likelihood, prior)
IP.UQ(exact=probInfo.exactSolution)

# %% Posterior
posterior = cuqi.distribution.Posterior(likelihood, prior)
cuqi.sampler.NUTS(posterior).sample(10,5); 

# %% Evaluations of likelihood and gradients are clear.
gt = probInfo.exactSolution
likelihood.logd(gt); # Gives value of log-likelihood function at x=gt.
likelihood.gradient(gt); #Gives gradient of log-likelihood function at x=gt.

# %% User Defined likelihood works similar to distributions, but makes clear its likelihood.
std = 0.2
f = lambda x: -1/(std**2)*np.sum((model@x-data)**2)
g = lambda x: -1/(std**2)*(model.T@(model@x - data))

likelihoodU = cuqi.likelihood.UserDefinedLikelihood(dim=n, logpdf_func=f, gradient_func=g)
likelihoodU

# %% Solvers
x0 = np.zeros(n)
x_MAP, info1 = cuqi.solver.maximize(posterior.logpdf, x0, posterior.gradient).solve()
x_ML, info2  = cuqi.solver.maximize(likelihood.logd, x0, likelihood.gradient).solve()
x_MLU, info3 = cuqi.solver.maximize(likelihoodU.logd, x0, likelihoodU.gradient).solve()

# %% pCN (likelihood+prior)
cuqi.sampler.pCN(posterior).sample_adapt(50); 
cuqi.sampler.pCN((likelihood,prior)).sample_adapt(50); 

# %% Likelihood with "hyperparameters"
likelihood = cuqi.distribution.Gaussian(mean=model, cov=lambda sigma: sigma**2).to_likelihood(data)
likelihood
