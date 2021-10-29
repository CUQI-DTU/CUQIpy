import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../..") 
import cuqi

# domain definition
N = 128           # spatial discretization
L = 1
T = 0.2
skip = 1

model = cuqi.model.Heat_1D(N=N, L=L, T=T, field_type='KL', skip=skip)
x = model.domain_geometry.grid
x_data = x[::skip]
M = x_data.shape[0]

# constructing signal
true_init = 100*x*np.exp(-5*x)*np.sin(L-x)

# defining the heat equation as the forward map
y_exact = model._advance_time(true_init) # observation vector

SNR = 200 # signal to noise ratio
sigma = np.linalg.norm(y_exact)/SNR
sigma2 = sigma*sigma # variance of the observation Gaussian noise

y_obs = y_exact + np.random.normal( 0, sigma, y_exact.shape )

likelihood = cuqi.distribution.GaussianCov(model, sigma2*np.eye(M))
prior = cuqi.distribution.GaussianCov(np.zeros(N), 1)
