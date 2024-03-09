import numpy as np
import cuqi
from cuqi.experimental.mcmc import SamplerNew
from cuqi.array import CUQIarray

class pCNNew(SamplerNew):  # Refactor to Proposal-based sampler?

    def __init__(self, target, scale=1.0, **kwargs):

        super().__init__(target, **kwargs)

        self.scale = scale
        self.current_point = self.initial_point
        self.current_loglike_eval = self._loglikelihood(self.current_point)

        self._acc = [1] # TODO. Check if we need this

    def validate_target(self):
        try:
            if isinstance(self.prior, (cuqi.distribution.Gaussian, cuqi.distribution.Normal)):
                pass
            else:
                raise ValueError("The prior distribution of the target need to be Gaussian")
        except AttributeError:
            raise ValueError("The target need to have a prior distribution")

    def step(self):
        # propose state
        xi = self.prior.sample(1).flatten()   # sample from the prior
        x_star = np.sqrt(1-self.scale**2)*self.current_point + self.scale*xi   # pCN proposal

        # evaluate target
        loglike_eval_star =  self._loglikelihood(x_star) 

        # ratio and acceptance probability
        ratio = loglike_eval_star - self.current_loglike_eval  # proposal is symmetric
        alpha = min(0, ratio)

        # accept/reject
        acc = 0
        u_theta = np.log(np.random.rand())
        if (u_theta <= alpha):
            self.current_point = x_star
            self.current_loglike_eval = loglike_eval_star
            acc = 1
        
        return acc

    @property
    def prior(self):
        if isinstance(self.target, cuqi.distribution.Posterior):
            return self.target.prior
        elif isinstance(self.target,tuple) and len(self.target)==2:
            return self.target[1]

    @property
    def likelihood(self):
        if isinstance(self.target, cuqi.distribution.Posterior):
            return self.target.likelihood
        elif isinstance(self.target,tuple) and len(self.target)==2:
            return self.target[0]

    @SamplerNew.target.setter 
    def target(self, value):
        if isinstance(value, cuqi.distribution.Posterior):
            self._target = value
            self._loglikelihood = lambda x : self.likelihood.logd(x)
        elif isinstance(value,tuple) and len(value)==2 and \
             (isinstance(value[0], cuqi.likelihood.Likelihood) or isinstance(value[0], cuqi.likelihood.UserDefinedLikelihood))  and \
             isinstance(value[1], cuqi.distribution.Distribution):
            self._target = value
            self._loglikelihood = lambda x : self.likelihood.logd(x)
        else:
            raise ValueError(f"To initialize an object of type {self.__class__}, 'target' need to be of type 'cuqi.distribution.Posterior'.")
        
        #TODO:
        #if not isinstance(self.prior,(cuqi.distribution.Gaussian, cuqi.distribution.Normal)):
        #    raise ValueError("The prior distribution of the target need to be Gaussian")

    @property
    def dim(self): # TODO. Check if we need this. Implemented in base class
        if hasattr(self,'target') and hasattr(self.target,'dim'):
            self._dim = self.target.dim
        elif hasattr(self,'target') and isinstance(self.target,tuple) and len(self.target)==2:
            self._dim = self.target[0].dim
        return self._dim

    def tune(self, skip_len, update_count):
        pass

    def get_state(self):
        return {'sampler_type': 'PCN', 'current_point': self.current_point.to_numpy(), \
                'current_loglike_eval': self.current_loglike_eval.to_numpy(), \
                'scale': self.scale}

    def set_state(self, state):
        temp = CUQIarray(state['current_point'] , geometry=self.target.geometry)
        self.current_point = temp
        temp = CUQIarray(state['current_loglike_eval'] , geometry=self.target.geometry)
        self.current_loglike_eval = temp
        self.scale = state['scale']
