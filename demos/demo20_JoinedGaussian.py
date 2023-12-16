# %%
import sys
sys.path.append("..") 
import cuqi
import numpy as np
import matplotlib.pyplot as plt

# %%
# Define two Gaussians on the entire domain
n = 128

z = np.zeros(n//4); o = np.ones(n//4) #Building blocks for defining means and covariances

X1 = cuqi.distribution.Gaussian(np.hstack((z,o,o,z)), prec=np.diag(np.hstack((100*o,2*o,2*o,100*o))))
X2 = cuqi.distribution.GMRF(np.hstack((o,o,o,o)),25,'zero')
# %%
plt.figure(); X1.sample(5).plot(); plt.title("X1 (Gaussian with high probability zeros in ends)")
plt.figure(); X2.sample(5).plot(); plt.title("X2 (GMRF)")

# %%
# First solve using only X1 "zero tails" prior
results_X1 = cuqi.testproblem.Deconvolution1D(prior=X1).sample_posterior(5000)

# %%
# Second solve using only GMRF prior
results_X2 = cuqi.testproblem.Deconvolution1D(prior=X2).sample_posterior(5000)

# %%
# Third, define joined Gaussian that incorporates high probability of zero in both ends.
means = [X1.mean,X2.mean]
sqrtprecs = [X1.sqrtprec,X2.sqrtprec.toarray()] #Either all sparse or all non-sparse
X = cuqi.distribution.JointGaussianSqrtPrec(means,sqrtprecs)

# and solve using joined prior
TP = cuqi.testproblem.Deconvolution1D(prior=X)
results_X = TP.sample_posterior(5000)

# %%
# Compare results
plt.figure(); results_X1.plot_ci(95,exact=TP.exactSolution); plt.title("Posterior with X1 (zero tail Gaussian)"); plt.ylim([-0.5,1.5])
plt.figure(); results_X2.plot_ci(95,exact=TP.exactSolution); plt.title("Posterior with X2 (GMRF)"); plt.ylim([-0.5,1.5])
plt.figure(); results_X.plot_ci(95,exact=TP.exactSolution); plt.title("Posterior with X (joined)"); plt.ylim([-0.5,1.5])
# %%
# Maximum a posteriori estimation (logpdf is not implemented, so throws an error.)
#x_MAP = TP.MAP()
