import numpy as np
import cuqi
from cuqi.experimental.mcmc import ProposalBasedSampler


class MH(ProposalBasedSampler):
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
        Additional keyword arguments to be passed to the base class :class:`ProposalBasedSampler`.

    """

    _STATE_KEYS = ProposalBasedSampler._STATE_KEYS.union({'scale', '_scale_temp'})

    def __init__(self, target=None, proposal=None, scale=1, **kwargs):
        super().__init__(target, proposal=proposal, scale=scale, **kwargs)

    def _initialize(self):
        # Due to a bug? in old MH, we must keep track of this extra variable to match behavior.
        self._scale_temp = self.scale

    def validate_target(self):
        # Fail only when there is no log density, which is currently assumed to be the case in case NaN is returned.
        if np.isnan(self.target.logd(self._get_default_initial_point(self.dim))):
            raise ValueError("Target does not have valid logd")

    def validate_proposal(self):
        if not isinstance(self.proposal, cuqi.distribution.Distribution):
            raise ValueError("Proposal must be a cuqi.distribution.Distribution object")
        if not self.proposal.is_symmetric:
            raise ValueError("Proposal must be symmetric")

    def step(self):
        # propose state
        xi = self.proposal.sample(1)   # sample from the proposal
        x_star = self.current_point + self.scale*xi.flatten()   # MH proposal

        # evaluate target
        target_eval_star = self.target.logd(x_star)

        # ratio and acceptance probability
        ratio = target_eval_star - self.current_target_logd # proposal is symmetric
        alpha = min(0, ratio)

        # accept/reject
        u_theta = np.log(np.random.rand())
        acc = 0
        if (u_theta <= alpha) and \
           (not np.isnan(target_eval_star)) and \
           (not np.isinf(target_eval_star)):
            self.current_point = x_star
            self.current_target_logd = target_eval_star
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
