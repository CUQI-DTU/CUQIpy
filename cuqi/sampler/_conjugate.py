from cuqi.distribution import Posterior, Gaussian, Gamma, ModifiedHalfNormal, GMRF
from cuqi.implicitprior import RegularizedGaussian, RegularizedGMRF, RegularizedUniform
from cuqi.geometry import Continuous1D, Continuous2D, Image2D
import numpy as np

class Conjugate: # TODO: Subclass from Sampler once updated
    """ Conjugate sampler

    Sampler for sampling a posterior distribution where the likelihood and prior are conjugate.

    Currently supported conjugate pairs are:
    - (Gaussian, Gamma)
    - (GMRF, Gamma)
    - (RegularizedGaussian, Gamma) with nonnegativity constraints only
    - (RegularizedGMRF, Gamma) with nonnegativity constraints only
    - (RegularizedUniform, Gamma) with l1 regularization only

    For more information on conjugate pairs, see https://en.wikipedia.org/wiki/Conjugate_prior.

    For implicit regularized Gaussians see:
    
    [1] Everink, Jasper M., Yiqiu Dong, and Martin S. Andersen. "Bayesian inference with projected densities." SIAM/ASA Journal on Uncertainty Quantification 11.3 (2023): 1025-1043.

    """
    # TODO: Update the documentation

    def __init__(self, target: Posterior):
        
        if not isinstance(target.likelihood.distribution, (Gaussian, GMRF, RegularizedGaussian, RegularizedGMRF, RegularizedUniform)):
            raise ValueError("Conjugate sampler only works with a Gaussian-type likelihood function")
        if not isinstance(target.prior, Gamma):
            raise ValueError("Conjugate sampler only works with Gamma prior")
        if not target.prior.dim == 1:
            raise ValueError("Conjugate sampler only works with univariate Gamma prior")
            
        if (isinstance(target.likelihood.distribution, (RegularizedGaussian, RegularizedGMRF)) and
            not isinstance(target.likelihood.distribution, (RegularizedUniform)) and
            target.likelihood.distribution.preset["constraint"] not in ["nonnegativity"] and
            target.likelihood.distribution.preset["regularization"] not in ["l1", "TV"]):
                raise ValueError("Conjugate sampler does not support the constraint and/or regularization options.")
        if (isinstance(target.likelihood.distribution, (RegularizedUniform)) and
            target.likelihood.distribution.preset["regularization"] not in ["l1", "TV"] and
            target.likelihood.distribution.preset["constraint"] not in [None, "nonnegativity"]):
                raise ValueError("Conjugate sampler does not support the constraint and/or regularization options.")
        
        self.target = target

    def step(self, x=None):

        if isinstance(self.target.likelihood.distribution, (Gaussian, GMRF)) and isinstance(self.target.prior, Gamma):
            return self._case_Gaussian_Gamma_prior()
        
        if isinstance(self.target.prior, Gamma):
            if isinstance(self.target.likelihood.distribution, RegularizedUniform):
                return self._case_RegularizedUniform_Gamma_prior()
            elif isinstance(self.target.likelihood.distribution, RegularizedGaussian) and self.target.likelihood.distribution.preset["regularization"] is None:
                return self._case_RegularizedUniform_Gamma_prior()
            
        if isinstance(self.target.likelihood.distribution, (RegularizedGaussian)) and isinstance(self.target.prior, (Gamma, ModifiedHalfNormal)):
            return self._case_RegularizedGaussian_ModifiedHalfNormal_prior()

    def _case_Gaussian_Gamma_prior(self):

        # Extract variables
        b = self.target.likelihood.data                                 #mu
        m = len(b)                                                      #n
        Ax = self.target.likelihood.distribution.mean                   #x_i
        L = self.target.likelihood.distribution(np.array([1])).sqrtprec #L

        alpha = self.target.prior.shape                                 #alpha
        beta = self.target.prior.rate                                   #beta

        base_rate = .5*np.linalg.norm(L@(Ax-b))**2

        # Create Gamma distribution and sample
        dist = Gamma(shape=m/2+alpha, rate=base_rate+beta)

        return dist.sample()
    
    def _case_RegularizedUniform_Gamma_prior(self):

        # Extract variables
        b = self.target.likelihood.data                                 #mu
        m = self._calc_m_for_RegularizedGaussians(b)                    #n
        Ax = self.target.likelihood.distribution.mean                   #x_i
        likelihood = self.target.likelihood.distribution(np.array([1]))
        L = likelihood.sqrtprec                                         #L

        alpha = self.target.prior.shape                                 #alpha
        beta = self.target.prior.rate                                   #beta

        base_rate = .5*np.linalg.norm(L@(Ax-b))**2

        if self.target.likelihood.distribution.preset["regularization"] == "l1":
            s = likelihood.strength[0]
            base_rate = s*np.linalg.norm(b, ord = 1)
        elif self.target.likelihood.distribution.preset["regularization"] == "TV":
            s = likelihood.strength[0]
            base_rate = s*np.linalg.norm(likelihood.transformation@b, ord = 1)
        else:
            base_rate = .5*np.linalg.norm(L@(Ax-b))**2

        base_shape = m
        if self.target.likelihood.distribution.preset["regularization"] is None:
            base_shape *= 0.5

        # Create Gamma distribution and sample
        dist = Gamma(shape=base_shape+alpha, rate=base_rate+beta)

        return dist.sample()
    
    def _case_RegularizedGaussian_ModifiedHalfNormal_prior(self):

        # Extract variables
        b = self.target.likelihood.data                                 #mu
        m = self._calc_m_for_RegularizedGaussians(b)                    #n
        Ax = self.target.likelihood.distribution.mean                   #x_i
        likelihood = self.target.likelihood.distribution(np.array([1]))
        L = likelihood.sqrtprec                                         #L

        # Turn Gamma into ModifiedHalfNormal
        if isinstance(self.target.prior, Gamma):
            alpha = self.target.prior.shape
            beta = self.target.prior.rate
            gamma = 0.0
        else:
            alpha = self.target.prior.alpha
            beta = self.target.prior.beta
            gamma = self.target.prior.gamma

        base_alpha = m
        base_beta = .5*np.linalg.norm(L@(Ax-b))**2

        if self.target.likelihood.distribution.preset["regularization"] == "l1":
            s = likelihood.strength[0]
            base_gamma = -s*np.linalg.norm(b, ord = 1)
        elif self.target.likelihood.distribution.preset["regularization"] == "TV":
            s = likelihood.strength[0]
            base_gamma = -s*np.linalg.norm(likelihood.transformation@b, ord = 1)

        # Create ModifiedHalfNormal distribution and sample
        dist = ModifiedHalfNormal(alpha = base_alpha + alpha,
                                  beta  = base_beta  + beta,
                                  gamma = base_gamma + gamma)

        return dist.sample()

    def _calc_m_for_RegularizedGaussians(self, b):
        """ Helper method to calculate m parameter for Gaussian-Gamma conjugate pair. """
        if isinstance(self.target.likelihood.distribution, (RegularizedGaussian, RegularizedGMRF)):
            threshold = 1e-6 # TODO: This could become a property of the class after the Conjugacy rework

            preset_constraint = self.target.likelihood.distribution.preset["constraint"]
            preset_regularization = self.target.likelihood.distribution.preset["regularization"]

            if preset_constraint == "nonnegativity" and preset_regularization is None:
                return np.count_nonzero(b) # Counting strict zeros, due to the solver used by RegularizedLinearRTO introducing actual zeros.
            
            if ((preset_constraint is None and preset_regularization == "l1") or
                (preset_constraint == "nonnegativity" and preset_regularization == "l1")):
                return Conjugate._count_weak_nonzero(b, threshold = threshold)
            
            if preset_constraint is None and preset_regularization == "TV":
                if isinstance(self.target.geometry, (Continuous1D)):
                    return Conjugate._count_weak_nonzero(self.target.likelihood.distribution.transformation@b, threshold = threshold)
                elif isinstance(self.target.geometry, (Continuous2D, Image2D)):
                    return Conjugate._count_weak_components_2d(b, threshold = threshold)
                else:
                    raise ValueError("Geometry not supported.")
            
            if preset_constraint == "nonnegativity" and preset_regularization == "TV":
                if isinstance(self.target.geometry, (Continuous1D)):
                    return Conjugate._count_weak_components_1D(b, threshold = threshold, lower = 0.0)
                elif isinstance(self.target.geometry, (Continuous2D, Image2D)):
                    return Conjugate._count_weak_components_2d(b, threshold = threshold, lower = 0.0)
                else:
                    raise ValueError("Geometry not supported.")
            
        raise Exception("Conjugacy pair not supported, although initial guards accepted it.")
    
    def _count_weak_nonzero(x, threshold = 1e-6):
        return np.sum([np.abs(v) >= threshold for v in x])
    
    def _count_weak_components_1D(x, threshold = 1e-6, lower = -np.inf, upper = np.inf):
        counter = 0
        if x[0] > lower and x[0] < upper:
            counter += 1
        
        x_previous = x[0]

        for x_current in x[1:]:
            if (abs(x_previous - x_current) >= threshold and
                x_current > lower and
                x_current < upper):
                    counter += 1

            x_previous = x_current
    
        return counter
        
    def _count_weak_components_2d(x, threshold = 1e-6, lower = -np.inf, upper = np.inf):
        filled = np.zeros_like(x, dtype = int)
        counter = 0

        def process(i, j):
            queue = []
            queue.append((i,j))
            filled[i, j] = 1
            while len(queue) != 0:
                (icur, jcur) = queue.pop(0)
                
                if icur > 0 and filled[icur - 1, jcur] == 0 and abs(x[icur, jcur] - x[icur - 1, jcur]) >= threshold:
                    filled[icur - 1, jcur] = 1
                    queue.append((icur-1, jcur))
                if jcur > 0 and filled[icur, jcur-1] == 0 and abs(x[icur, jcur] - x[icur, jcur - 1]) >= threshold:
                    filled[icur, jcur-1] = 1
                    queue.append((icur, jcur-1))
                if icur < x.shape[0]-1 and filled[icur + 1, jcur] == 0 and abs(x[icur, jcur] - x[icur + 1, jcur]) >= threshold:
                    filled[icur + 1, jcur] = 1
                    queue.append((icur+1, jcur))
                if jcur < x.shape[1]-1 and filled[icur, jcur + 1] == 0 and abs(x[icur, jcur] - x[icur, jcur + 1]) >= threshold:
                    filled[icur, jcur + 1] = 1
                    queue.append((icur, jcur+1))
        
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if filled[i,j] == 0:
                    if x[i,j] > lower and x[i,j] < upper:
                        counter += 1
                    process(i, j)
        return counter
                    
