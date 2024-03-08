import numpy as np
import cuqi
from cuqi.mcmc import ProposalBasedSamplerNew
from cuqi.array import CUQIarray

class CWMHNew(ProposalBasedSamplerNew):
    """Component-wise Metropolis Hastings sampler.

    Allows sampling of a target distribution by a component-wise random-walk sampling of a proposal distribution along with an accept/reject step.

    Parameters
    ----------

    target : `cuqi.distribution.Distribution` or lambda function
        The target distribution to sample. Custom logpdfs are supported by using a :class:`cuqi.distribution.UserDefinedDistribution`.
    
    proposal : `cuqi.distribution.Distribution` or callable method
        The proposal to sample from. If a callable method it should provide a single independent sample from proposal distribution. Defaults to a Gaussian proposal.  *Optional*.

    scale : float
        Scale parameter used to define correlation between previous and proposed sample in random-walk.  *Optional*.

    x0 : ndarray
        Initial parameters. *Optional*

    dim : int
        Dimension of parameter space. Required if target and proposal are callable functions. *Optional*.

    callback : callable, *Optional*
        If set this function will be called after every sample.
        The signature of the callback function is `callback(sample, sample_index)`,
        where `sample` is the current sample and `sample_index` is the index of the sample.
        An example is shown in demos/demo31_callback.py.

    Example
    -------
    .. code-block:: python

        # Parameters
        dim = 5 # Dimension of distribution
        mu = np.arange(dim) # Mean of Gaussian
        std = 1 # standard deviation of Gaussian

        # Logpdf function
        logpdf_func = lambda x: -1/(std**2)*np.sum((x-mu)**2)

        # Define distribution from logpdf as UserDefinedDistribution (sample and gradients also supported as inputs to UserDefinedDistribution)
        target = cuqi.distribution.UserDefinedDistribution(dim=dim, logpdf_func=logpdf_func)

        # Set up sampler
        sampler = cuqi.sampler.CWMH(target, scale=1)

        # Sample
        samples = sampler.sample(2000)

    """
    def __init__(self, target: cuqi.density.Density, proposal=None, scale=1,
                 initial_point=None, **kwargs):
        super().__init__(target, proposal=proposal, scale=scale,
                         initial_point=initial_point, **kwargs)
        self._acc = [np.ones((self.dim))]

    def validate_target(self):
        pass # All targets are valid

    @ProposalBasedSamplerNew.proposal.setter 
    def proposal(self, value):
        fail_msg = "Proposal should be either None, "+\
            f"{cuqi.distribution.Distribution.__class__.__name__} "+\
            "conditioned only on 'location' and 'scale', lambda function, "+\
            f"or {cuqi.distribution.Normal.__class__.__name__} conditioned "+\
            "only on 'mean' and 'std'"

        if value is None:
            self._proposal = cuqi.distribution.Normal(mean = lambda location:location,std = lambda scale:scale, geometry=self.dim)

        elif isinstance(value, cuqi.distribution.Distribution) and sorted(value.get_conditioning_variables())==['location','scale']:
            self._proposal = value

        elif isinstance(value, cuqi.distribution.Normal) and sorted(value.get_conditioning_variables())==['mean','std']:
            self._proposal = value(mean = lambda location:location, std = lambda scale:scale)

        elif not isinstance(value, cuqi.distribution.Distribution) and callable(value):
            self._proposal = value

        else:
            raise ValueError(fail_msg)
        
        #self._proposal.geometry = self.target.geometry

    def step(self): #CWMH_new
        # Propose state x_i_star used to update x_t
        # each component of x_t step by step 
        x_t = self.current_point
        target_eval_t = self.current_target
        if isinstance(self.proposal,cuqi.distribution.Distribution):
            x_i_star = self.proposal(location= x_t, scale = self.scale).sample()
        else:
            x_i_star = self.proposal(x_t, self.scale) 
        x_star = x_t.copy()
        acc = np.zeros(self.dim)

        for j in range(self.dim):
            # propose state
            x_star[j] = x_i_star[j]

            # evaluate target
            target_eval_star = self.target.logd(x_star)

            # ratio and acceptance probability
            ratio = target_eval_star - target_eval_t  # proposal is symmetric
            alpha = min(0, ratio)

            # accept/reject
            u_theta = np.log(np.random.rand())
            if (u_theta <= alpha):
                x_t[j] = x_i_star[j]
                target_eval_t = target_eval_star
                acc[j] = 1
            else:
                pass
                # x_t[j]       = x_t[j]
                # target_eval_t = target_eval_t
            x_star = x_t.copy()
        self.current_point = x_t.copy()
        self.current_target = target_eval_t
        #NEW: update return 
        #return x_t, target_eval_t, acc
        return acc

    def tune(self, skip_len, update_count):
        star_acc = 0.21/self.dim + 0.23
        hat_acc = np.mean(self._acc[-1-skip_len:], axis=0)

        # compute new scaling parameter
        zeta = 1/np.sqrt(update_count+1)   # ensures that the variation of lambda(i) vanishes
        scale_temp = np.exp(
            np.log(self.scale) + zeta*(hat_acc-star_acc))  

        # update parameters
        self.scale = np.minimum(scale_temp, np.ones(self.dim))

    def get_state(self):
        return {'sampler_type': 'CWMH', 'current_point': self.current_point.to_numpy(), 'current_target': self.current_target, 'scale': self.scale}

    def set_state(self, state):
        self.current_point =\
            CUQIarray(state['current_point'] , geometry=self.target.geometry)

        self.current_target = state['current_target']

        self.scale = state['scale']