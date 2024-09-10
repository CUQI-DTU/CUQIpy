import numpy as np
from cuqi.distribution import Distribution
from cuqi.distribution import Gaussian

class UserDefinedDistribution(Distribution):
    """
    Class to wrap user-defined logpdf, gradient, and/or sampling callable into CUQIpy Distribution.

    Parameters
    ------------
    logpdf_func: Function evaluating log probability density function. Callable.
    gradient_func: Function evaluating the gradient of the logpdf. Callable.
    sample_func: Function drawing samples from distribution. Callable.
    
    Example
    -----------
    .. code-block:: python

        # Generate an i.i.d. n-dim Gaussian with zero mean and 2 variance.
        mu1 = -1.0
        std1 = 4.0
        X = cuqi.distribution.Normal(mean=mu1, std=std1)
        dim1 = 1
        logpdf_func = lambda xx: -np.log(std1*np.sqrt(2*np.pi))-0.5*((xx-mu1)/std1)**2
        sample_func = lambda : mu1 + std1*np.random.randn(dim1,1)
        XU = cuqi.distribution.UserDefinedDistribution(dim=dim1, logpdf_func=logpdf_func, sample_func=sample_func)
    """

    def __init__(self, dim=None, logpdf_func=None, gradient_func=None, sample_func=None, **kwargs):

        # Init from abstract distribution class
        super().__init__(**kwargs)

        if logpdf_func is not None and not callable(logpdf_func): raise ValueError("logpdf_func should be callable.")
        if sample_func is not None and not callable(sample_func): raise ValueError("sample_func should be callable.")
        if gradient_func is not None and not callable(gradient_func): raise ValueError("grad_func should be callable.")
        
        self.geometry = dim # Store dimension by calling setter for geometry (creates default geometry)
        self.logpdf_func = logpdf_func
        self.sample_func = sample_func
        self.gradient_func = gradient_func

    def logpdf(self, x):
        if self.logpdf_func is not None:
            return self.logpdf_func(x)
        else:
            raise NotImplementedError("logpdf_func is not defined.")
    
    def _gradient(self, x):
        if self.gradient_func is not None:
            return self.gradient_func(x)
        else:
            raise NotImplementedError("gradient_func is not defined.")

    def _sample(self, N=1, rng=None):
        #TODO(nabr) allow sampling more than 1 sample and potentially rng?
        if self.sample_func is not None:
            if N==1:
                return self.sample_func().flatten()
            else:
                out = np.zeros((self.dim,N))
                for i in range(N):
                    out[:,i] = self.sample_func()
                return out
        else:
            raise Exception("sample_func is not defined. Sampling can be performed by passing the density to a sampler from the 'sampler' module.")

    @property
    def _mutable_vars(self):
        """ Returns the mutable variables of the distribution. """
        # Currently mutable variables are not supported for user-defined distributions.
        return []

    def get_conditioning_variables(self):
        """ Returns the conditioning variables of the distribution. """
        # Currently conditioning variables are not supported for user-defined distributions.
        return []

### BENCHMARKS
class DistributionGallery(UserDefinedDistribution):
    # we follow the benchmarks from:
    # https://github.com/chi-feng/mcmc-demo/blob/master/main/MCMC.js

    def __init__(self, distribution_name,**kwargs):
        # Init from abstract distribution class
        if distribution_name == "CalSom91":
            #TODO: user can specify sig and delta
            dim = 2
            self.sig = 0.1
            self.delta = 1
            self.dfdx1 = lambda x1, x2: -(x1*(np.sqrt(x2**2+x1**2)-1))/(self.sig**2*np.sqrt(x2**2+x1**2))
            self.dfdx2 = lambda x1, x2: -(x2*(np.sqrt(x2**2+x1**2)-1))/(self.sig**2*np.sqrt(x2**2+x1**2))-(x2-1)/self.delta**2
            logpdf_func = self._CalSom91_logpdf_func
            grad_logpdf = self._CalSom91_grad_logpdf
        elif distribution_name == "BivariateGaussian":
            #TODO: user can specify Gaussain input
            #TODO: Keep Gaussian distribution other functionalities (e.g. _sample)
            dim = 2
            mu = np.zeros(dim)
            sigma = np.diag(np.linspace(0.5, 1, dim))
            R = np.array([[1.0, 0.9 ],[0.9, 1.0]])
            dist = Gaussian(mu, sigma@R@sigma)
            self._sample = dist._sample
            logpdf_func = dist.logpdf
            grad_logpdf = dist.gradient
        elif distribution_name == "funnel":
            # "funnel" distribution from Neal (2003) - Slice Sampling. Annals of Statistics 31(3): 705-67
            dim = 2
            self.m0, self.m1 = 0, 0
            self.s1 = 3
            self.f = lambda x, m, s: -0.5*np.log(2*np.pi) - np.log(s) - 0.5*((x-m)/s)**2
            self.dfdx = lambda x, m, s: -(x-m)/(s**2)
            self.dfds = lambda x, m, s: -1/s + ((x-m)**2)/(s**3)
            #
            logpdf_func = self._funnel_logpdf_func
            grad_logpdf = self._funnel_grad_logpdf
        elif distribution_name == "mixture":
            dim = 2
            self.m0, self.m1, self.m2 = np.array([-1.5, -1.5]), np.array([1.5, 1.5]), np.array([-2, 2])
            self.S0, self.S1, self.S2 = (0.8**2), (0.8**2), (0.5**2)
            self.G0 = Gaussian(self.m0, self.S0)
            self.G1 = Gaussian(self.m1, self.S1)
            self.G2 = Gaussian(self.m2, self.S2)
            # self.grad_logG = lambda x, mu, s: -np.diag([1/s, 1/s])@(x-mu).T
            #
            logpdf_func = self._mixture_logpdf_func
            grad_logpdf = self._mixture_grad_func
        elif distribution_name == "squiggle":
            dim = 2
            m0 = np.zeros(dim)
            S0 = np.array([[2, 0.25], [0.25, 0.5]])
            self.G0 = Gaussian(m0, S0)
            #
            logpdf_func = self._squiggle_logpdf_func
            grad_logpdf = self._squiggle_grad_logpdf
        elif distribution_name == "donut":
            dim = 2
            self.radius, self.sigma2 = 2.6, 0.033
            #
            logpdf_func = self._donut_logpdf_func
            grad_logpdf = self._donut_grad_logpdf
        elif distribution_name == "banana":
            dim = 2
            m0 = np.array([0, 4])
            S0 = np.array([[1, 0.5], [0.5, 1]])
            self.G0 = Gaussian(m0, S0)
            self.a, self.b = 2, 0.2
            #
            logpdf_func = self._banana_logpdf_func
            grad_logpdf = self._banana_grad_logpdf

        super().__init__(logpdf_func=logpdf_func, gradient_func=grad_logpdf, dim=dim, **kwargs)

    def _CalSom91_logpdf_func(self, x):
        if len(x.shape) == 1:
            x = x.reshape((1, 2))
        return -1/(2*self.sig**2)*(np.sqrt(x[:,0]**2+ x[:,1]**2) -1 )**2 -1/(2*self.delta**2)*(x[:,1]-1)**2

    def _CalSom91_grad_logpdf(self, x):
        if len(x.shape) == 1:
            x = x.reshape((1, 2))
        grad = np.array([self.dfdx1(x[:, 0], x[:, 1]), self.dfdx2(x[:, 0], x[:, 1])])
        if x.shape[0] == 1:
            return grad.flatten()
        else:
            return grad

    def _funnel_logpdf_func(self, x):
        if len(x.shape) == 1:
            x = x.reshape((1, 2))
        s0 = np.exp(x[:, 1]/2)
        return self.f(x[:, 0], self.m0, s0) + self.f(x[:, 1], self.m1, self.s1)

    def _funnel_grad_logpdf(self, x):
        if len(x.shape) == 1:
            x = x.reshape((1, 2))
        s0 = np.exp(x[:, 1]/2)
        grad = np.array([self.dfdx(x[:, 0], self.m0, s0), \
                        self.dfds(x[:, 0], self.m0, s0)*0.5*s0 + self.dfdx(x[:, 1], self.m1, self.s1)])
        if x.shape[0] == 1:
            return grad.flatten()
        else:
            return grad

    def _mixture_logpdf_func(self, x):
        if len(x.shape) == 1:
            x = x.reshape((1, 2))
        return np.log(self.G0.pdf(x) + self.G1.pdf(x) + self.G2.pdf(x))
    
    def _mixture_grad_func(self, x):
        if len(x.shape) == 1:
            x = x.reshape((1, 2))
        p1, p2, p3 = self.G0.pdf(x), self.G1.pdf(x), self.G2.pdf(x)
        scale = np.nan_to_num(1 / (p1 + p2 + p3))
        grad = (p1*self.G0.gradient(x) + p2*self.G1.gradient(x) + p3*self.G2.gradient(x)) * scale
        if x.shape[0] == 1:
            return grad.flatten()
        else:
            return grad

    def _squiggle_logpdf_func(self, x):
        if len(x.shape) == 1:
            x = x.reshape((1, 2))
        y = np.zeros((x.shape[0], self.dim))
        y[:, 0], y[:, 1] = x[:, 0], x[:, 1] + np.sin(5*x[:, 0])
        return self.G0.logpdf(y)

    def _squiggle_grad_logpdf(self, x):
        if len(x.shape) == 1:
            x = x.reshape((1, 2))
        y = np.zeros((x.shape[0], self.dim))
        y[:, 0], y[:, 1] = x[:, 0], x[:, 1] + np.sin(5*x[:, 0])
        grad = self.G0.gradient(y)
        gradx0, gradx1 = grad[0, :] + grad[1, :]*5*np.cos(5*x[:, 0]), grad[1, :]
        grad[0, :], grad[1, :] = gradx0, gradx1
        if x.shape[0] == 1:
            return grad.flatten()
        else:
            return grad

    def _donut_logpdf_func(self, x):
        if len(x.shape) == 1:
            x = x.reshape((1, 2))
        r = np.linalg.norm(x, axis=1)
        return - (r - self.radius)**2 / self.sigma2

    def _donut_grad_logpdf(self, x):
        if len(x.shape) == 1:
            x = x.reshape((1, 2))
        r = np.linalg.norm(x, axis=1)
        idx = np.argwhere(r==0)
        r[idx] = 1e-16
        grad = np.array([(x[:, 0]*((self.radius/r)-1)*2)/self.sigma2, \
                         (x[:, 1]*((self.radius/r)-1)*2)/self.sigma2])
        if x.shape[0] == 1:
            return grad.flatten()
        else:
            return grad

    def _banana_logpdf_func(self, x):
        if len(x.shape) == 1:
            x = x.reshape((1, 2))
        y = np.zeros((x.shape[0], self.dim))
        y[:, 0] = x[:, 0]/self.a
        y[:, 1] = x[:, 1]*self.a + self.a*self.b*(x[:, 0]**2 + self.a**2)
        return self.G0.logpdf(y)

    def _banana_grad_logpdf(self, x):
        if len(x.shape) == 1:
            x = x.reshape((1, 2))
        y = np.zeros((x.shape[0], self.dim))
        y[:, 0] = x[:, 0]/self.a
        y[:, 1] = x[:, 1]*self.a + self.a*self.b*(x[:, 0]**2 + self.a**2)
        grad = self.G0.gradient(y)
        gradx0, gradx1 = grad[0, :]/self.a + grad[1, :]*self.a*self.b*2*x[:, 0], grad[1, :]*self.a
        grad[0, :], grad[1, :] = gradx0, gradx1
        if x.shape[0] == 1:
            return grad.flatten()
        else:
            return grad
