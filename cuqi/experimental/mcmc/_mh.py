import numpy as np
import cuqi
from cuqi.experimental.mcmc import ProposalBasedSamplerNew
from cuqi.array import CUQIarray


class MHNew(ProposalBasedSamplerNew):
    """ Metropolis-Hastings (MH) sampler.

    Parameters
    ----------
    target : cuqi.density.Density
        Target density or distribution.

    proposal : cuqi.distribution.Distribution or callable
        Proposal distribution. If None, a random walk MH is used (i.e., Gaussian proposal with identity covariance).

    scale : float
        Scaling parameter for the proposal distribution.

    kwargs : dict
        Additional keyword arguments to be passed to the base class :class:`ProposalBasedSamplerNew`.

    """

    def __init__(self, target, proposal=None, scale=1, **kwargs):
        super().__init__(target, proposal=proposal, scale=scale, **kwargs)
        # Due to a bug? in old MH, we must keep track of this extra variable to match behavior.
        self._scale_temp = self.scale 

    def validate_target(self):
        pass # All targets are valid

    @ProposalBasedSamplerNew.proposal.setter  # TODO. Check if we can refactor this. We can work with a validate_proposal method instead?
    def proposal(self, value):
        fail_msg = "Proposal should be either None, symmetric cuqi.distribution.Distribution or a lambda function."

        if value is None:
            self._proposal = cuqi.distribution.Gaussian(np.zeros(self.dim), 1)
        elif not isinstance(value, cuqi.distribution.Distribution) and callable(value):
            raise NotImplementedError(fail_msg)
        elif isinstance(value, cuqi.distribution.Distribution) and value.is_symmetric:
            self._proposal = value
        else:
            raise ValueError(fail_msg)
        self._proposal.geometry = self.target.geometry

    def step(self):
        # propose state
        xi = self.proposal.sample(1)   # sample from the proposal
        x_star = self.current_point + self.scale*xi.flatten()   # MH proposal

        # evaluate target
        target_eval_star = self.target.logd(x_star)

        # ratio and acceptance probability
        ratio = target_eval_star - self.current_target # proposal is symmetric
        alpha = min(0, ratio)

        # accept/reject
        u_theta = np.log(np.random.rand())
        acc = 0
        if (u_theta <= alpha):
            self.current_point = x_star
            self.current_target = target_eval_star
            acc = 1
        
        return acc

    def tune(self, skip_len, update_count):
        hat_acc = np.mean(self._acc[-skip_len:])

        # d. compute new scaling parameter
        zeta = 1/np.sqrt(update_count+1)   # ensures that the variation of lambda(i) vanishes

        # We use self._scale_temp here instead of self.scale in update. This might be a bug,
        # but is equivalent to old MH
        self._scale_temp = np.exp(np.log(self._scale_temp) + zeta*(hat_acc-0.234))

        # update parameters
        self.scale = min(self._scale_temp, 1)

    def get_state(self):
        return {'sampler_type': 'MH', 'current_point': self.current_point.to_numpy(), 'current_target': self.current_target.to_numpy(), 'scale': self.scale}

    def set_state(self, state):
        temp = CUQIarray(state['current_point'] , geometry=self.target.geometry)
        self.current_point = temp
        temp = CUQIarray(state['current_target'] , geometry=self.target.geometry)
        self.current_target = temp
        self.scale = state['scale']
