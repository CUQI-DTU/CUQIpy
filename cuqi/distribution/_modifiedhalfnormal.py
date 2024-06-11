import numpy as np
import scipy.stats as sps
import scipy.special as special
from cuqi.distribution import Distribution
from cuqi.utilities import force_ndarray

class ModifiedHalfNormal(Distribution):
    """
    Represents a modified half-normal (MHN) distribution, a three-parameter family of distributions generalizing the Gamma distribution.
    The distribution is continuous with pdf 
    f(x; alpha, beta, gamma) propto x^(alpha-1) * exp(-beta * x^2 + gamma * x)

    The MHN generalizes the half-normal distribution, because
    f(x; 1, beta, 0) propto exp(-beta * x^2)

    The MHN generalizes the gamma distribution because
    f(x; alpha, 0, -gamma) propto x^(alpha-1) * exp(- gamma * x)

    Reference:
    [1] Sun, et al. "The Modified-Half-Normal distribution: Properties and an efficient sampling scheme." Communications in Statistics-Theory and Methods
    
    Parameters
    ----------
    alpha : float
        The polynomial exponent parameter of the MHN distribution. Must be positive.

    beta : float
        The quadratic exponential parameter of the MHN distribution. Must be positive.

    gamma : float
        The linear exponential parameter of the MHN distribution.

    """
    def __init__(self, alpha, beta, gamma, is_symmetric=False, **kwargs):
        # Init from abstract distribution class
        super().__init__(is_symmetric=is_symmetric,**kwargs) 

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def logpdf(self, x): # Unnormalized
        return (self.alpha - 1)*np.log(x) - self.beta * x * x + self.gamma * x


    def _MHN_sample_gamma_proposal(self, alpha, beta, gamma, delta = None):
        if delta is None:
            delta = beta + (gamma*gamma - gamma*np.sqrt(gamma*gamma + 8*beta*alpha))/(4*alpha)
            
        while True:
            T = np.random.gamma(alpha/2, 1.0/delta)
            X = np.sqrt(T)
            U = np.random.uniform()
            if X > 0 and np.log(U) < -(beta-delta)*T + gamma*X - gamma*gamma/(4*(beta-delta)):
                return X
            
    def _MHN_sample_normal_proposal(self, alpha, beta, gamma, mu):
        if mu is None:
            mu = (gamma + np.sqrt(gamma*gamma + 8*beta*(alpha - 1)))/(4*beta)    
        
        while True:
            X = np.random.normal(mu, np.sqrt(0.5/beta))
            U = np.random.uniform()
            if X > 0 and np.log(U) < (alpha-1)*np.log(X) - np.log(mu) + (2*beta*mu-gamma)*(mu-X):
                return X

    # Sample from MHN when alpha > 1, beta > 0 and gamma > 0  (Algorithm 1 from [1])
    def _MHN_sample_positive_gamma_1(self, alpha, beta, gamma):
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
            return self._MHN_sample_normal_proposal(alpha, beta, gamma, mu)
        else: # Use sqrt(gamma) proposal
            return self._MHN_sample_gamma_proposal(alpha, beta, gamma, delta)


    # Sample from MHN when alpha > 0, beta > 0 and gamma <= 0 (Algorithm 3 from [1])
    # ""
    def _MHN_sample_negative_gamma(self, alpha, beta, gamma, m = None):
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
            T = np.random.gamma(alpha*val1, 1.0/val2)
            X = m*np.power(T,val1)
            U = np.random.uniform()
            if np.log(U) < val2*T-beta*X*X+gamma*X:
                return X
            

    """
    Sample from distribution with density proportional to
    x^(alpha - 1)*exp*(-beta*x^2 + gamma*x)
    """
    def _MHN_sample(self, alpha, beta, gamma, m = None):
        if gamma <= 0.0:
            return self._MHN_sample_negative_gamma(alpha, beta, gamma, m = m)
        
        if alpha >= 1:
            return self._MHN_sample_positive_gamma_1(alpha, beta, gamma)
        
        return self._MHN_sample_gamma_proposal(alpha, beta, gamma)

    def _sample(self, N, rng=None):
        return np.array([self._MHN_sample(self.alpha, self.beta, self.gamma,) for _ in range(N)])
            