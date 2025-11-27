import numpy as np
import scipy.stats as sps
import scipy.special as special
from cuqi.distribution import Distribution
from cuqi.utilities import force_ndarray

class ModifiedHalfNormal(Distribution):
    """
    Represents a modified half-normal (MHN) distribution, a three-parameter family of distributions generalizing the Gamma distribution.
    The distribution is continuous with pdf

    .. math::
    
        f(x; \\alpha, \\beta, \\gamma) \propto x^{(\\alpha-1)} * \exp(-\\beta * x^2 + \\gamma * x)

    The MHN generalizes the half-normal distribution, because
    :math:`f(x; 1, \\beta, 0) \propto \exp(-\\beta * x^2)`

    The MHN generalizes the gamma distribution because
    :math:`f(x; \\alpha, 0, -\\gamma) \propto x^{(\\alpha-1)} * \exp(- \\gamma * x)`

    Reference:
    [1] Sun, et al. "The Modified-Half-Normal distribution: Properties and an efficient sampling scheme." Communications in Statistics-Theory and Methods
    
    Parameters
    ----------
    alpha : float or array_like
        The polynomial exponent parameter :math:`\\alpha` of the MHN distribution. Must be positive.

    beta : float or array_like
        The quadratic exponential parameter :math:`\\beta` of the MHN distribution. Must be positive.

    gamma : float or array_like
        The linear exponential parameter :math:`\\gamma` of the MHN distribution.

    """
    def __init__(self, alpha=None, beta=None, gamma=None, is_symmetric=False, **kwargs):
        # Init from abstract distribution class
        super().__init__(is_symmetric=is_symmetric, **kwargs) 

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    @property
    def alpha(self):
        """ The polynomial exponent parameter of the MHN distribution. Must be positive. """
        return self._alpha
    
    @alpha.setter
    def alpha(self, value):
        self._alpha = force_ndarray(value, flatten=True)

    @property
    def beta(self):
        """ The quadratic exponential parameter of the MHN distribution. Must be positive. """
        return self._beta
    
    @beta.setter
    def beta(self, value):
        self._beta = force_ndarray(value, flatten=True)

    @property
    def gamma(self):
        """ The linear exponential parameter of the MHN distribution. """
        return self._gamma
    
    @gamma.setter
    def gamma(self, value):
        self._gamma = force_ndarray(value, flatten=True)

    def logpdf(self, x): # Unnormalized
        return np.sum((self.alpha - 1)*np.log(x) - self.beta * x * x + self.gamma * x)

    def _gradient_scalar(self, val):
        if val <= 0.0:
            return np.nan
        return (self.alpha - 1)/val - 2*self.beta*val + self.gamma

    def _gradient(self, val, *args, **kwargs):
        return np.array([self._gradient_scalar(v) for v in val])

    def _MHN_sample_gamma_proposal(self, alpha, beta, gamma, rng, delta=None):
        """
            Sample from a modified half-normal distribution using a Gamma distribution proposal.
        """
        if delta is None:
            delta = beta + (gamma*gamma - gamma*np.sqrt(gamma*gamma + 8*beta*alpha))/(4*alpha)
            
        while True:
            T = rng.gamma(alpha/2, 1.0/delta)
            X = np.sqrt(T)
            U = rng.uniform()
            if X > 0 and np.log(U) < -(beta-delta)*T + gamma*X - gamma*gamma/(4*(beta-delta)):
                return X
            
    def _MHN_sample_normal_proposal(self, alpha, beta, gamma, mu, rng):
        """
            Sample from a modified half-normal distribution using a Normal/Gaussian distribution proposal.
        """
        if mu is None:
            mu = (gamma + np.sqrt(gamma*gamma + 8*beta*(alpha - 1)))/(4*beta)    
        
        while True:
            X = rng.normal(mu, np.sqrt(0.5/beta))
            U = rng.uniform()
            if X > 0 and np.log(U) < (alpha-1)*np.log(X) - np.log(mu) + (2*beta*mu-gamma)*(mu-X):
                return X

    def _MHN_sample_positive_gamma_1(self, alpha, beta, gamma, rng):
        """
            Sample from a modified half-normal distribution, assuming alpha is greater than one and gamma is positive.
        """
        if gamma <= 0.0:
            raise ValueError("gamma needs to be positive")
            
        if alpha <= 1.0:
            raise ValueError("alpha needs to be greater than 1.0")

        # Decide whether to use Normal or sqrt(Gamma) proposals for acceptance-rejectance scheme
        mu = (gamma + np.sqrt(gamma*gamma + 8*beta*(alpha - 1)))/(4*beta)
        K1  = 2*np.sqrt(np.pi)
        K1 *= np.power((np.sqrt(beta)*(alpha-1))/(2*beta*mu-gamma), alpha - 1)
        K1 *= np.exp(-(alpha-1)+beta*mu*mu)
        
        delta = beta + (gamma*gamma - gamma*np.sqrt(gamma*gamma + 8*beta*alpha))/(4*alpha)
        K2  = np.power(beta/delta, 0.5*alpha)
        K2 *= special.gamma(alpha/2.0)
        K2 *= np.exp(gamma*gamma/(4*(beta-delta)))

        if K2 > K1: # Use normal proposal
            return self._MHN_sample_normal_proposal(alpha, beta, gamma, mu, rng)
        else: # Use sqrt(gamma) proposal
            return self._MHN_sample_gamma_proposal(alpha, beta, gamma, rng, delta)

    def _MHN_sample_negative_gamma(self, alpha, beta, gamma, rng, m=None):
        """
            Sample from a modified half-normal distribution, assuming gamma is negative.
            The argument 'm' is the matching point, see Algorithm 3 from [1] for details.
        """
        if gamma > 0.0:
            raise ValueError("gamma needs to be negative")
            
        if m is None:
            if alpha <= 1.0:
                m = 1.0
            else:
                m = "mode"
        
        # The acceptance rate of this choice is at least 0.5*sqrt(2) approx 70.7 percent, according to Theorem 4 from [1].
        if isinstance(m, str) and m.lower() == "mode":
            m = (gamma + np.sqrt(gamma*gamma + 8*beta*alpha))/(4*beta)

        while True:
            val1 = (beta*m-gamma)/(2*beta*m-gamma)
            val2 = m*(beta*m-gamma)
            T = rng.gamma(alpha*val1, 1.0/val2)
            X = m*np.power(T,val1)
            U = rng.uniform()
            if np.log(U) < val2*T-beta*X*X+gamma*X:
                return X
            
    def _MHN_sample(self, alpha, beta, gamma, m=None, rng=None):
        """
        Sample from a modified half-normal distribution using an algorithm from [1].
        """
        if rng == None:
            rng = np.random

        if gamma <= 0.0:
            return self._MHN_sample_negative_gamma(alpha, beta, gamma, m=m, rng=rng)
        
        if alpha > 1:
            return self._MHN_sample_positive_gamma_1(alpha, beta, gamma, rng=rng)
        
        return self._MHN_sample_gamma_proposal(alpha, beta, gamma, rng=rng)

    def _sample(self, N, rng=None):
        if hasattr(self.alpha, '__getitem__'):
            return np.array([[self._MHN_sample(self.alpha[i], self.beta[i], self.gamma[i], rng=rng) for i in range(len(self.alpha))] for _ in range(N)])
        else:
            return np.array([self._MHN_sample(self.alpha, self.beta, self.gamma, rng=rng) for i in range(N)])

            