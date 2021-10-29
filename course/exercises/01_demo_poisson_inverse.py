import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../..") 
import cuqi

N = 129            # spatial discretization 
L = 1
skip = 4

# random field parameters
mean = 0.0
var = 10
lc = 0.2
p = 2
C_YY = lambda x1, x2: var*np.exp( -(1/p) * (abs(x1-x2)/lc)**p )
KL_map = lambda x: np.exp(x)
f = lambda xs: 10*np.exp( -( (xs - 0.5)**2 ) / 0.02) # source term
d_KL = 50
model = cuqi.model.Poisson_1D(N=N, L=L, source=f, skip=skip ,field_type='CustomKL', cov_fun=C_YY, mean=mean, std=np.sqrt(var), d_KL=d_KL, KL_map=KL_map)
x = model.domain_geometry.grid
x_data = x[::skip]
M = x_data.shape[0]

# constructing signal
true_kappa = np.exp( 5*x*np.exp(-2*x)*np.sin(L-x) )
y_exact = model._solve_with_conductivity( true_kappa )


SNR = 200 # signal to noise ratio
sigma = np.linalg.norm(y_exact)/SNR
sigma2 = sigma*sigma # variance of the observation Gaussian noise
y_obs = y_exact + np.random.normal( 0, sigma, y_exact.shape )

likelihood = cuqi.distribution.GaussianCov(model, sigma2*np.eye(M))
prior = cuqi.distribution.GaussianCov(np.zeros(N), 1)