# %% Initialize and import CUQI
import sys
sys.path.append("..") 
import numpy as np
import cuqi
from cuqi.model import LinearModel
from cuqi.distribution import Gaussian, LMRF, CMRF, Gamma
from cuqi.problem import BayesianProblem

# %% Minimal example

# Import data and forward matrix
Amat   = np.load("data/Deconvolution.npz")["A"]          #Matrix (numpy)
y_data = np.load("data/Deconvolution.npz")["data"]       #Vector (numpy)
m,n    = Amat.shape

# Data from square function
y_data = cuqi.testproblem.Deconvolution1D(phantom="square").data
x_exact = cuqi.testproblem.Deconvolution1D(phantom="square").exactSolution

# Set up Bayesian model for inverse problem
A  = LinearModel(Amat)                          # y = Ax. Model for inverse problem
x  = Gaussian(np.zeros(n), 0.1)                 # x ~ N(0,0.1)
y  = Gaussian(A@x, 0.05)                        # y ~ N(Ax,0.05)
IP = BayesianProblem(y, x).set_data(y=y_data)   # Bayesian problem given observed data
samples = IP.UQ(exact=x_exact)                  # Run UQ analysis

# %%
# Wrap into CUQI "testproblem".
TP = cuqi.testproblem.Deconvolution1D(prior=x, phantom="square")
TP.UQ()

# %%
# switch prior
# Set up Bayesian model for inverse problem
A  = LinearModel(Amat)                                  # y = Ax. Model for inverse problem
x  = LMRF(0, 0.01, geometry=n, bc_type='zero')               # x ~ LMRF(0, 0.01), Zero BC
y  = Gaussian(A@x, 0.05)                                # y ~ N(Ax,0.05)
IP = BayesianProblem(y, x).set_data(y=y_data)           # Bayesian problem given observed data
samples = IP.UQ(exact=x_exact)                          # Run UQ analysis

# %%
# switch prior again
# Set up Bayesian model for inverse problem
A  = LinearModel(Amat)                                  # y = Ax. Model for inverse problem
x  = CMRF(0, 0.01, geometry=n, bc_type='zero')               # x ~ CMRF(0,0.01), Zero BC
y  = Gaussian(A@x, 0.05)                                # y ~ N(Ax,0.05)
IP = BayesianProblem(y, x).set_data(y=y_data)           # Bayesian problem given observed data
samples = IP.UQ(exact=x_exact)                          # Run UQ analysis

# %% Hierarchical Bayesian models
# Set up Bayesian model for inverse problem,
# now with hyper-parameter on noise precision!
A  = LinearModel(Amat)                                   # y = Ax. Model for inverse problem
l  = Gamma(1, 1e-2)                                      # l ~ Gamma(1, 10^-2)
x  = Gaussian(np.zeros(n), 0.1)                          # x ~ N(0, 0.1)
y  = Gaussian(A@x, lambda l: 1/l)                        # y ~ N(Ax, l^-1)
IP = BayesianProblem(y, x, l).set_data(y=y_data)         # Bayesian problem given observed data
samples = IP.UQ(Ns = 2000, exact={"x":x_exact, "l":400}) # Run UQ analysis

# %% Hierarchical Bayesian models
# Set up Bayesian model for inverse problem,
# now with hyper-parameters on noise precision and prior precision.
A  = LinearModel(Amat)                                   # y = Ax. Model for inverse problem
d  = Gamma(1, 1e-2)                                      # d ~ Gamma(1, 10^-2)
l  = Gamma(1, 1e-2)                                      # l ~ Gamma(1, 10^-2)
x  = Gaussian(np.zeros(n), cov=lambda d: 1/d)            # x ~ N(0, d^-1)
y  = Gaussian(A@x, cov=lambda l: 1/l)                    # y ~ N(Ax, l^-1)
IP = BayesianProblem(y, x, d, l).set_data(y=y_data)      # Bayesian problem given observed data
samples = IP.UQ(Ns = 2000, exact={"x":x_exact, "l":400}) # Run UQ analysis

# %% Hierarchical Bayesian models (switch prior)
# Set up Bayesian model for inverse problem with a different prior
A  = LinearModel(Amat)                                   # y = Ax. Model for inverse problem
d  = Gamma(1, 1e-2)                                      # d ~ Gamma(1, 10^-2)
l  = Gamma(1, 1e-2)                                      # l ~ Gamma(1, 10^-2)
x  = LMRF(0, lambda d: 1/d, geometry=n)                  # x ~ LMRF(0, d^{-1}), Zero BC
y  = Gaussian(A@x, cov=lambda l: 1/l)                    # y ~ N(Ax, l^-1)
IP = BayesianProblem(y, x, d, l).set_data(y=y_data)      # Bayesian problem given observed data
samples = IP.UQ(Ns = 1000, exact={"x":x_exact, "l":400}) # Run UQ analysis

# %% Hierarchical Bayesian models (Not implemented choices)
try:
    A  = LinearModel(Amat)                                   # y = Ax. Model for inverse problem
    d  = Gamma(1, 1e-2)                                      # d ~ Gamma(1, 10^-2)
    l  = Gamma(1, 1e-2)                                      # l ~ Gamma(1, 10^-2)
    x  = CMRF(0, lambda d: 1/d, geometry=n)                  # x ~ CMRF(0, d^{-1}), Zero BC
    y  = Gaussian(A@x, cov=lambda l: 1/l)                    # y ~ N(Ax, l^-1)
    IP = BayesianProblem(y, x, d, l).set_data(y=y_data)      # Bayesian problem given observed data
    samples = IP.UQ(Ns = 1000, exact={"x":x_exact, "l":400}) # Run UQ analysis
except NotImplementedError as e:
    print(e)

# %%
