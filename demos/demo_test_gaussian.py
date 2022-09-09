# %%
import sys
sys.path.append("..")

import numpy as np
import scipy as sp

import cuqi

dim = 5
prec = sp.sparse.diags([1, -4, 6, -4, 1], [-2, -1, 0, 1, 2], shape=(5, 5))
cov = sp.linalg.inv(prec.toarray())
sqrtprec = sp.linalg.cholesky(prec.toarray())

X_prec = cuqi.distribution.Gaussian(np.zeros(dim), prec=prec.toarray())
X_cov = cuqi.distribution.Gaussian(np.zeros(dim), cov)
X_sqrtprec = cuqi.distribution.Gaussian(np.zeros(dim), sqrtprec=sqrtprec)
X_GMRF = cuqi.distribution.GMRF(np.zeros(dim), 1, 1, 'zero', order=2)

#
x0 = np.random.randn(dim)
a = np.allclose(X_cov.logpdf(x0), X_GMRF.logpdf(x0))
b = np.allclose(X_cov.logpdf(x0), X_sqrtprec.logpdf(x0)) #TODO: Returns complex number
c = np.allclose(X_cov.logpdf(x0), X_prec.logpdf(x0))
d = np.allclose(X_cov.gradient(x0), X_GMRF.gradient(x0))
e = np.allclose(X_cov.gradient(x0), X_prec.gradient(x0))

def stats(samples):
    """ Compute meadian, std, and lo95, up95 of samples """
    return np.vstack((np.median(samples, axis=1), np.std(samples, axis=1), np.percentile(samples, 2.5, axis=1), np.percentile(samples, 97.5, axis=1)))

Ns = 10000
s_cov = stats(X_cov.sample(Ns).samples)
s_GMRF = stats(X_GMRF.sample(Ns).samples)
s_sqrtprec = stats(X_sqrtprec.sample(Ns).samples)
s_prec = stats(X_prec.sample(Ns).samples)

f = np.allclose(np.round(s_cov, 1), np.round(s_GMRF, 1) , rtol=0.1)
g = np.allclose(np.round(s_cov, 1), np.round(s_sqrtprec, 1) , rtol=0.1)
h = np.allclose(np.round(s_cov, 1), np.round(s_prec, 1) , rtol=0.1)

X_prec_s = cuqi.distribution.Gaussian(np.zeros(dim), prec=prec)
X_sqprec = cuqi.distribution.Gaussian(np.zeros(dim), sqrtprec=sqrtprec)

cov_s = sp.sparse.linalg.inv(prec)
sqrtprec_s = sp.sparse.csr_matrix(sqrtprec)
X_cov_s = cuqi.distribution.Gaussian(np.zeros(dim), cov=cov_s)
X_sqprec_s = cuqi.distribution.Gaussian(np.zeros(dim), sqrtprec=sqrtprec_s)

i = np.allclose(X_cov._logupdf(x0), X_cov_s._logupdf(x0))
j =  np.allclose(X_prec._logupdf(x0), X_prec_s._logupdf(x0))
k =  np.allclose(X_sqprec._logupdf(x0), X_sqprec_s._logupdf(x0))

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
