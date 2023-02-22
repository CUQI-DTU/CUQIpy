from cuqi.distribution import Posterior, Gaussian, GMRF, Gamma, InverseGamma, UserDefinedDistribution
import numpy as np
import scipy.sparse as spa
import matplotlib.pyplot as plt

try:
    import sksparse.cholmod as skchol
    has_cholmod = True
except ImportError:
    has_cholmod = False
eps = 1e-15
epsm = 1e13

class Conjugate: # TODO: Subclass from Sampler once updated
    """ Conjugate sampler

    Sampler for sampling a posterior distribution where the likelihood and prior are conjugate.

    Currently supported conjugate pairs are:
    - (Gaussian, Gaussian)
    - (Gaussian, Gamma and Invgamma)
    - (Invgamma, Invgamma)

    For more information on conjugate pairs, see https://en.wikipedia.org/wiki/Conjugate_prior.

    """

    def __init__(self, target: Posterior):
        if not isinstance(target.likelihood.distribution, (Gaussian, GMRF, InverseGamma)):
            raise ValueError("Conjugate sampler only works with a Gaussian and inverse-gamma conditionals")
        if not isinstance(target.prior, (Gaussian, GMRF, Gamma, InverseGamma)):
            raise ValueError("Conjugate sampler only works with Gaussian, Gamma and InverseGamma priors")
        self.target = target

    def step(self, x=None):
        # Extract variables
        prior = self.target.prior
        likelihood = self.target.likelihood
        
        # Create closed-form conditionals and sample them
        if isinstance(likelihood.distribution, (Gaussian, GMRF)) and isinstance(prior, (Gaussian, GMRF)):
            # extract hyperprior/prior parameters
            mu_pr = prior.mean
            Prec_pr = prior.sqrtprec.T @ prior.sqrtprec
            n_pr = prior.dim

            # extract conditional/likelihood parameters
            A = likelihood.model._matrix
            ATA = spa.csc_matrix(A.T @ A)
            y = likelihood.data
            Ax = likelihood.distribution.mean
            Prec_like = likelihood.distribution(np.ones(n_pr)).prec

            # density is prop to a Gaussian
            Prec_pos = spa.csc_matrix(Prec_like*ATA + Prec_pr)
            L_chol = skchol.cholesky(Prec_pos, ordering_method='natural')
            mu_pos = L_chol(Prec_like*(A.T@y) + Prec_pr@mu_pr)
            # dist = Gaussian(mu_pos, prec=Prec_pos)
            u = np.random.randn(n_pr)
            samplefunc = lambda: mu_pos + L_chol.solve_Lt(u, use_LDLt_decomposition=False) 
            dist = UserDefinedDistribution(sample_func=samplefunc)

        elif isinstance(likelihood.distribution, (Gaussian, GMRF)) and isinstance(prior, (InverseGamma)):
            # extract hyperprior/prior parameters
            alpha_pr = prior.shape
            beta_pr = prior.scale
            n_pr = prior.dim

            # extract conditional/likelihood parameters
            x = likelihood.data # plt.plot(x)
            n = len(x)
            mu = likelihood.distribution.mean
            L = likelihood.distribution(np.ones(n_pr)).sqrtprec

            # density is prop to an inverse gamma
            if n_pr == 1:
                dist = InverseGamma(shape=alpha_pr+(n/2), location=0, scale=beta_pr+0.5*np.linalg.norm(L@(x-mu))**2)
            else:
                dist = InverseGamma(shape=alpha_pr+(1/2), location=0, scale=beta_pr+0.5*(L@(x-mu))**2)

        elif isinstance(likelihood.distribution, (Gaussian, GMRF)) and isinstance(prior, (Gamma)):
            b = likelihood.data                                 #mu
            m = len(b)                                          #n
            Ax = likelihood.distribution.mean                   #x_i
            L = likelihood.distribution(np.array([1])).sqrtprec #L
            #
            alpha = prior.shape                                 #alpha
            beta = prior.rate                                   #beta
            dist = Gamma(shape=m/2+alpha, rate=0.5*np.linalg.norm(L@(b-Ax))**2 + beta)

        elif isinstance(likelihood.distribution, (InverseGamma)) and isinstance(prior, (InverseGamma)):
            # Eq. 4.13 Horseshoe paper: scale parameters given as beta_pr + nu/'data'
            # extract hyperprior/prior parameters
            alpha_pr = prior.shape
            beta_pr = prior.scale
            n_pr = prior.dim

            # extract conditional/likelihood parameters
            alpha_like = likelihood.distribution.shape
            nu_like = likelihood.distribution(np.ones(n_pr)).scale

            # compute the conditional parameters
            alpha_cond = alpha_pr + alpha_like
            beta_cond = beta_pr + nu_like/likelihood.data

            # density is prop to an inverse gamma
            dist = InverseGamma(shape=alpha_cond, location=0, scale=beta_cond)

        return dist.sample()