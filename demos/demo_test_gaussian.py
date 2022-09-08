# %%
import sys
sys.path.append("..")

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import cuqi

prec = sp.sparse.diags([1, -4, 6, -4, 1], [-2, -1, 0, 1, 2], shape=(5, 5))
cov = sp.linalg.inv(prec.toarray())
sqrtprec = sp.linalg.cholesky(prec.toarray())
X_prec = cuqi.distribution.Gaussian(np.zeros(5), prec=prec)
X_cov = cuqi.distribution.Gaussian(np.zeros(5), cov)
X_sqrtprec = cuqi.distribution.Gaussian(np.zeros(5), sqrtprec=sqrtprec)
    
# =====TEST COV=========================
# Create sparse, symmetric PSD matrix S
# n = 10
# mean = np.zeros(n)
# A = np.random.randn(n, n)  # Unit normal gaussian distribution.
# A[sp.sparse.rand(n, n, 0.85).todense().nonzero()] = 0  # Sparsen the matrix.
# Strue = A.dot(A.T) + 0.05 * np.eye(n)  # Force strict pos. def.
# cov = sp.sparse.csr_matrix(np.linalg.inv(Strue))

# prior = cuqi.distribution.Gaussian(mean, cov)
# x0 = 1000*np.random.rand(prior.dim)
# eval1 = prior.logpdf(x0)
# eval2 = sp.stats.multivariate_normal.logpdf(x0, mean, cov.todense())
# xstar = prior.sample(1)
# np.allclose(eval1,eval2)

# =======TEST PREC=======================
# n = 100
# mean = np.zeros(n)
# A = np.random.randn(n, n)  # Unit normal gaussian distribution.
# A[sp.sparse.rand(n, n, 0.85).todense().nonzero()] = 0  # Sparsen the matrix.
# Strue = A.dot(A.T) + 0.05 * np.eye(n)  # Force strict pos. def.
# cov = np.linalg.inv(Strue) # sp.sparse.csr_matrix(
# prior = cuqi.distribution.Gaussian(mean, prec=Strue)
# x0 = 1000*np.random.rand(prior.dim)
# eval1 = prior.logpdf(x0)
# eval2 = sp.stats.multivariate_normal.logpdf(x0, mean, cov)
# xstar = prior.sample(1)
# np.allclose(eval1,eval2)

# =======TEST 1D=======================
mean = 0
cov = 4
sqrtcov = 2
sqrtprec = 1/sqrtcov
prec = 1/cov

prior = cuqi.distribution.Gaussian(mean, sqrtprec=sqrtprec)
x0 = 10*np.random.rand(prior.dim)
eval1 = prior.logpdf(x0)
eval2 = sp.stats.norm.logpdf(x0, mean, sqrtcov)
xstar = prior.sample(1000).samples
np.allclose(eval1,eval2)

# =======TEST SQRTCOV=======================
n = 100
mean = np.zeros(n)
A = np.random.randn(n, n)  # Unit normal gaussian distribution.
A[sp.sparse.rand(n, n, 0.85).todense().nonzero()] = 0  # Sparsen the matrix.
Prec = A.dot(A.T) + 0.05 * np.eye(n)  # Force strict pos. def.
Cov = np.linalg.inv(Prec) # sp.sparse.csr_matrix()
sqrtCov = np.linalg.cholesky(Cov)

prior = cuqi.distribution.Gaussian(mean, sqrtcov=sqrtCov)
x0 = 1000*np.random.rand(prior.dim)
eval1 = prior.logpdf(x0)
eval2 = sp.stats.multivariate_normal.logpdf(x0, mean, Cov)
xstar = prior.sample(1)
np.allclose(eval1,eval2)

# =======TEST SQRTPREC=======================
n = 100
mean = np.zeros(n)
A = np.random.randn(n, n)  # Unit normal gaussian distribution.
A[sp.sparse.rand(n, n, 0.85).todense().nonzero()] = 0  # Sparsen the matrix.
Prec = A.dot(A.T) + 0.05 * np.eye(n)  # Force strict pos. def.
Cov = np.linalg.inv(Prec) # sp.sparse.csr_matrix()
sqrtPrec = sp.sparse.csr_matrix(np.linalg.cholesky(Prec))

prior = cuqi.distribution.Gaussian(mean, sqrtprec=2)
x0 = 1000*np.random.rand(prior.dim)
eval1 = prior.logpdf(x0)
eval2 = sp.stats.multivariate_normal.logpdf(x0, mean, np.diag(np.ones(n)*0.25))
xstar = prior.sample(1)
np.allclose(eval1,eval2)
