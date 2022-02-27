# %% Import CUQI and other packages
import sys
sys.path.append("..")
import cuqi
import matplotlib.pyplot as plt
import numpy as np

from cuqi.testproblem import Deconvolution
from cuqi.distribution import Gaussian
from cuqi.problem import BayesianProblem

# Parameters
n = 128

# %% Likelihood is a function defined according to a conditional distribution.
# X ~ p(x | theta): Conditional distribution.
# L(theta | x=10) : p(x=10 | theta): The likelihood function w. "data" x=10.
X = cuqi.distribution.Normal(mean=0, std=lambda theta: theta**2)
L = X.to_likelihood(data=10)
print(X)
print(L)

# %% Five line example (likelihood + prior) Q: do we also keep BP(data_dist, prior, data)?
model, data, probInfo = Deconvolution.get_components(dim=n, phantom="Square")
likelihood = Gaussian(mean=model, std=0.05).to_likelihood(data)
prior = Gaussian(mean=np.zeros(n), std=0.2)
IP = BayesianProblem(likelihood, prior)
IP.UQ(exact=probInfo.exactSolution)

# %% Posterior (conditional on data)
data_dist = Gaussian(mean=model, std=0.2)
prior_dist = Gaussian(mean=np.zeros(n), std=0.2)
post_dist_cond = cuqi.distribution.Posterior(data_dist, prior_dist)
print(post_dist_cond)

# Adding data creates a fully defined distribution
post_dist = post_dist_cond(data=data)
print(post_dist)

# Q: Should we allow providing likelihood and prior to posterior like below?
post_dist_v2 = cuqi.distribution.Posterior(likelihood, prior) #Does not work as intended.

# Q: Should we get rid of the "conditional" idea for posterior and just use likelihood, prior?

# %% Evaluations of likelihood and gradients are clear.
gt = probInfo.exactSolution
likelihood(x=gt); # Gives value of log-likelihood function at x=gt.
likelihood.gradient(x=gt); #Gives gradient of log-likelihood function at x=gt.

# %% User Defined likelihood works similar to distributions, but makes clear its likelihood.
std = 0.2
f = lambda x: -1/(std**2)*np.sum((model@x-data)**2)
g = lambda x: -1/(std**2)*(model.T@(model@x - data))

likelihoodU = cuqi.likelihood.UserDefinedLikelihood(dim=n, logpdf_func=f, gradient_func=g)
likelihoodU

# %% Solvers
x0 = np.zeros(n)
x_MAP, info1 = cuqi.solver.maximize(post_dist.logpdf, x0, post_dist.gradient).solve()
#x_ML, info2  = cuqi.solver.maximize(likelihood, x0, likelihood.gradient).solve()
x_MLU, info3 = cuqi.solver.maximize(likelihoodU, x0, likelihoodU.gradient).solve()

# %% Cons to work at
# 1: Likelihood, prior was not given to BayesianProblem (it was data_dist, prior, data).
#    This is now updated to only be likelihood, prior (see above)

# 2: “Likelihood(x)+prior.logpdf(x)“. Different interface for likelihood and prior.
#    If likelihood can only be likelihood function this is OK I believe. (Check w. pCN).

# 3: pdf, cdf are gone from likelihood.
#    Solution: They still exist in "likelihood.distribution".
#    Also, they should not be part of the likelihood function!

# 4: "Conditional likelihood".
#    Solution: There is no concept of conditional likelihood anymore. Just a function.
#    Note: If we need to distinguish between "x" and other params later like "sigma",
#    we can use this: Gaussian(mean=model, lambda sigma: sigma**2, conds=["sigma"]).

# 5: Maximizer(posterior.logpdf), Maximizer(likelihood), Maximizer(prior.logpdf)
#    Posterior and prior act differently than likelhood.
#    Note: This is by design now.. We can discuss if this is good design though..
#    Idea: We could add a logpdf method? And write that its strictly not the PDF,
#    but convenient for the code.

# 6: cuqi.likelihood.Likelihood(data_dist, data)..
#    Note: Its mostly data_dist.to_likelihood(data) now. User will most likely use this approach.
