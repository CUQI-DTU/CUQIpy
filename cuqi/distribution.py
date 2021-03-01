import numpy as np
import scipy.stats as sps
from scipy.special import erf, loggamma, gammainc
from scipy.sparse import spdiags, eye
from scipy.linalg import eigh


# ========================================================================
class Cauchy_diff(object):

    def __init__(self, location, scale, bndcond):
        self.loc = location
        self.scale = scale
        self.dim = len(location)
        self.bnd = bndcond

        # finite difference matrix
        one_vec = np.ones(self.dim)
        diags = np.vstack([-one_vec, one_vec])
        if (bndcond == 'zero'):
            locs = [-1, 0]
            Dmat = spdiags(diags, locs, self.dim+1, self.dim)
        elif (bndcond == 'periodic'):
            locs = [-1, 0]
            Dmat = spdiags(diags, locs, self.dim+1, self.dim).tocsr()
            Dmat[-1, 0] = 1
            Dmat[0, -1] = -1
        elif (bndcond == 'neumann'):
            locs = [0, 1]
            Dmat = spdiags(diags, locs, self.dim-1, self.dim)
        elif (bndcond == 'backward'):
            locs = [0, -1]
            Dmat = spdiags(diags, locs, self.dim, self.dim).tocsr()
            Dmat[0, 0] = 1
        elif (bndcond == 'none'):
            Dmat = eye(self.dim)
        self.D = Dmat

    def pdf(self, x):
        Dx = self.D @ (x-self.loc)
        return (1/(np.pi**len(Dx))) * np.prod(self.scale/(Dx**2 + self.scale**2))

    def logpdf(self, x):
        Dx = self.D @ (x-self.loc)
        # g_logpr = (-2*Dx/(Dx**2 + gamma**2)) @ D
        return -len(Dx)*np.log(np.pi) + sum(np.log(self.scale) - np.log(Dx**2 + self.scale**2))
    
    # def cdf(self, x):   # TODO
    #     return 1/np.pi * np.atan((x-self.loc)/self.scale)

    # def sample(self):   # TODO
    #     return self.loc + self.scale*np.tan(np.pi*(np.random.rand(self.dim)-1/2))


# ========================================================================
class Normal(object):

    def __init__(self, mean, std, dim=1):
        self.mean = mean
        self.std = std
        self.dim = dim

    def pdf(self, x):
        return 1/(self.std*np.sqrt(2*np.pi))*np.exp(-0.5*((x-self.mean)/self.std)**2)

    def logpdf(self, x):
        return -np.log(self.std*np.sqrt(2*np.pi))-0.5*((x-self.mean)/self.std)**2

    def cdf(self, x):
        return 0.5*(1 + erf((x-self.mean)/(self.std*np.sqrt(2))))

    def sample(self):
        return np.random.normal(self.mean, self.std, self.dim)


# ========================================================================
class Gamma(object):

    def __init__(self, shape, rate):
        self.shape = shape
        self.rate = rate
        self.scale = 1/rate

    def pdf(self, x):
        # sps.gamma.pdf(x, a=self.shape, loc=0, scale=self.scale)
        # (self.rate**self.shape)/(gamma(self.shape)) * (x**(self.shape-1)*np.exp(-self.rate*x))
        return np.exp(self.logpdf(x))

    def logpdf(self, x):
        # sps.gamma.logpdf(x, a=self.shape, loc=0, scale=self.scale)
        return (self.shape*np.log(self.rate)-loggamma(self.shape)) + ((self.shape-1)*np.log(x) - self.rate*x)

    def cdf(self, x):
        # sps.gamma.cdf(x, a=self.shape, loc=0, scale=self.scale)
        return gammainc(self.shape, self.rate*x)

    def sample(self, N):
        return np.random.gamma(shape=self.shape, scale=self.scale, size=(N))


# ========================================================================
class Gaussian(object):

    def __init__(self, mean, std, corrmat):
        self.mean = mean
        self.std = std
        self.R = corrmat
        self.dim = len(np.diag(corrmat))
        #self = sps.multivariate_normal(mean, (std**2)*corrmat)

        # pre-computations (covariance and determinants)
        if isinstance(std, (list, tuple, np.ndarray)):
            self.Sigma = np.diag(std) @ (corrmat @ np.diag(std))   # covariance
            isdiag = np.count_nonzero(corrmat - np.diag(np.diagonal(corrmat)))
            if (isdiag == 0):    # uncorrelated
                self.det = np.prod(std**2)
                self.logdet = sum(2*np.log(std))
                self.L = np.linalg.cholesky(self.Sigma)
            else:
                self.det = np.linalg.det(self.Sigma)
                self.L = np.linalg.cholesky(self.Sigma)
                self.logdet = 2*sum(np.log(np.diag(self.L)))  # only for PSD matrices
        else:
            self.Sigma = np.diag(std*np.ones(self.dim)) @ (corrmat @ np.diag(std*np.ones(self.dim)))   # covariance
            isdiag = np.count_nonzero(corrmat - np.diag(np.diagonal(corrmat)))
            if (isdiag == 0):   # uncorrelated
                self.det = std**(2*self.dim)
                self.logdet = 2*self.dim*np.log(std)
                self.L = np.linalg.cholesky(self.Sigma)
            else:
                self.det = std**(2*self.dim) * np.linalg.det(corrmat)
                self.L = np.linalg.cholesky(self.Sigma)
                self.logdet = 2*sum(np.log(np.diag(self.L)))  # only for PSD matrices

        # inverse of Cholesky
        self.Linv = np.linalg.inv(self.L)

        # Compute decomposition such that Q = U @ U.T
        # self.Q = np.linalg.inv(self.Sigma)   # precision matrix
        # s, u = eigh(self.Q, lower=True, check_finite=True)
        # s_pinv = np.array([0 if abs(x) <= 1e-5 else 1/x for x in s], dtype=float)
        # self.U = u @ np.diag(np.sqrt(s_pinv))

    def logpdf(self, x1, *x2):
        if callable(self.mean):
            mu = self.mean(x2[0])   # mean is variable
        else:
            mu = self.mean       # mean is fix
        xLinv = (x1 - mu) @ self.Linv.T
        quadform = np.sum(np.square(xLinv), 1) if (len(xLinv.shape) > 1) else np.sum(np.square(xLinv))
        # = sps.multivariate_normal.logpdf(x1, mu, self.Sigma)
        return -0.5*(self.logdet + quadform + self.dim*np.log(2*np.pi))

    def pdf(self, x1, *x2):
        # = sps.multivariate_normal.pdf(x1, self.mean, self.Sigma)
        return np.exp(self.logpdf(x1, *x2))

    def cdf(self, x1):   # TODO
        return sps.multivariate_normal.cdf(x1, self.mean, self.Sigma)

    def sample(self, N):   # TODO
        return np.reshape(np.random.multivariate_normal(self.mean, self.Sigma, N),[self.dim,N])


# ========================================================================
class Laplace_diff(object):

    def __init__(self, location, scale, bndcond):
        self.loc = location
        self.scale = scale
        self.dim = len(location)
        self.bnd = bndcond

        # finite difference matrix
        one_vec = np.ones(self.dim)
        diags = np.vstack([-one_vec, one_vec])
        if (bndcond == 'zero'):
            locs = [-1, 0]
            Dmat = spdiags(diags, locs, self.dim+1, self.dim)
        elif (bndcond == 'periodic'):
            locs = [-1, 0]
            Dmat = spdiags(diags, locs, self.dim+1, self.dim).tocsr()
            Dmat[-1, 0] = 1
            Dmat[0, -1] = -1
        elif (bndcond == 'neumann'):
            locs = [0, 1]
            Dmat = spdiags(diags, locs, self.dim-1, self.dim)
        elif (bndcond == 'backward'):
            locs = [0, -1]
            Dmat = spdiags(diags, locs, self.dim, self.dim).tocsr()
            Dmat[0, 0] = 1
        elif (bndcond == 'none'):
            Dmat = eye(self.dim)
        self.D = Dmat

    def pdf(self, x):
        Dx = self.D @ (x-self.loc)  # np.diff(X)
        return (1/(2*self.scale))**(len(Dx)) * np.exp(-np.linalg.norm(Dx, ord=1, axis=0)/self.scale)

    def logpdf(self, x):
        Dx = self.D @ (x-self.loc)
        return len(Dx)*(-(np.log(2)+np.log(self.scale))) - np.linalg.norm(Dx, ord=1, axis=0)/self.scale

    # def cdf(self, x):   # TODO
    #     return 1/2 + 1/2*np.sign(x-self.loc)*(1-np.exp(-np.linalg.norm(x, ord=1, axis=0)/self.scale))

    # def sample(self):   # TODO
    #     p = np.random.rand(self.dim)
    #     return self.loc - self.scale*np.sign(p-1/2)*np.log(1-2*abs(p-1/2))


class GMRF(object):
    def __init__(self):
        raise NotImplementedError
      
class Uniform(object):
    def __init__(self):
        raise NotImplementedError