import numpy as np
import cuqi
from cuqi.experimental.mcmc import Sampler
from cuqi.array import CUQIarray

class PCN(Sampler):  # Refactor to Proposal-based sampler?

    _STATE_KEYS = Sampler._STATE_KEYS.union({'scale', 'current_likelihood_logd', 'lambd'})

    def __init__(self, target=None, scale=1.0, **kwargs):

        super().__init__(target, **kwargs)
        self.initial_scale = scale

    def _initialize(self):
        self.scale = self.initial_scale
        self.current_likelihood_logd = self._loglikelihood(self.current_point)

        # parameters used in the Robbins-Monro recursion for tuning the scale parameter
        # see details and reference in the tune method
        self.lambd = self.scale
        self.star_acc = 0.44 #TODO: 0.234 # target acceptance rate

    def validate_target(self):
        if not isinstance(self.target, cuqi.distribution.Posterior):
            raise ValueError(f"To initialize an object of type {self.__class__}, 'target' need to be of type 'cuqi.distribution.Posterior'.")
        if not isinstance(self.prior, (cuqi.distribution.Gaussian, cuqi.distribution.Normal)):
            raise ValueError("The prior distribution of the target need to be Gaussian")

    def step(self):
        # propose state
        xi = self.prior.sample(1).flatten()   # sample from the prior
        x_star = np.sqrt(1-self.scale**2)*self.current_point + self.scale*xi   # PCN proposal

        # evaluate target
        loglike_eval_star =  self._loglikelihood(x_star) 

        # ratio and acceptance probability
        ratio = loglike_eval_star - self.current_likelihood_logd  # proposal is symmetric
        alpha = min(0, ratio)

        # accept/reject
        acc = 0
        u_theta = np.log(np.random.rand())
        if (u_theta <= alpha):
            self.current_point = x_star
            self.current_likelihood_logd = loglike_eval_star
            acc = 1
        
        return acc

    @property
    def prior(self):
        return self.target.prior

    @property
    def likelihood(self):
        return self.target.likelihood
        
    def _loglikelihood(self, x):
        return self.likelihood.logd(x)

    @property
    def dim(self): # TODO. Check if we need this. Implemented in base class
        if hasattr(self,'target') and hasattr(self.target,'dim'):
            self._dim = self.target.dim
        elif hasattr(self,'target') and isinstance(self.target,tuple) and len(self.target)==2:
            self._dim = self.target[0].dim
        return self._dim

    def tune(self, skip_len, update_count):
        """
        Tune the scale parameter of the PCN sampler.
        The tuning is based on algorithm 4 in Andrieu, Christophe, and Johannes Thoms. 
        "A tutorial on adaptive MCMC." Statistics and computing 18 (2008): 343-373.
        Note: the tuning algorithm here is the same as the one used in MH sampler.
        """

        # average acceptance rate in the past skip_len iterations
        hat_acc = np.mean(self._acc[-skip_len:])

        # new scaling parameter zeta to be used in the Robbins-Monro recursion
        zeta = 1/np.sqrt(update_count+1)

        # Robbins-Monro recursion to ensure that the variation of lambd vanishes
        self.lambd = np.exp(np.log(self.lambd) + zeta*(hat_acc-self.star_acc))

        # update scale parameter
        self.scale = min(self.lambd, 1)
