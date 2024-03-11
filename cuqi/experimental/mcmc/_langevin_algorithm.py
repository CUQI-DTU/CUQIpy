import numpy as np
import cuqi
from cuqi.experimental.mcmc import SamplerNew
from cuqi.array import CUQIarray

class MALANew(SamplerNew): # Refactor to Proposal-based sampler?

    def __init__(self, target, scale=1.0, **kwargs):

        super().__init__(target, **kwargs)

        self.scale = scale
        self.current_point = self.initial_point
        self.current_target_eval = self.target.logd(self.current_point)
        self.current_target_grad_eval = self.target.gradient(self.current_point)
        self._acc = [1] # TODO. Check if we need this

    def validate_target(self):
        try:
            self.target.gradient(np.ones(self.dim))
            pass
        except (NotImplementedError, AttributeError):
            raise ValueError("The target need to have a gradient method")

    def step(self):
        # propose state
        xi = cuqi.distribution.Normal(mean=np.zeros(self.dim), std=np.sqrt(self.scale)).sample()
        x_star = self.current_point + 0.5*self.scale*self.current_target_grad_eval + xi

        # evaluate target
        target_eval_star, target_grad_star = self.target.logd(x_star), self.target.gradient(x_star)

        # Metropolis step
        log_target_ratio = target_eval_star - self.current_target_eval
        log_prop_ratio = self.log_proposal(self.current_point, x_star, target_grad_star) \
            - self.log_proposal(x_star, self.current_point,  self.current_target_grad_eval)
        log_alpha = min(0, log_target_ratio + log_prop_ratio)

        # accept/reject
        acc = 0
        log_u = np.log(cuqi.distribution.Uniform(low=0, high=1).sample())
        if (log_u <= log_alpha) and (np.isnan(target_eval_star) == False):
            self.current_point = x_star
            self.current_target_eval = target_eval_star
            self.current_target_grad_eval = target_grad_star
            acc = 1

        return acc

    def tune(self, skip_len, update_count):
        pass

    def log_proposal(self, theta_star, theta_k, g_logpi_k):
        mu = theta_k + ((self.scale)/2)*g_logpi_k
        misfit = theta_star - mu
        return -0.5*((1/(self.scale))*(misfit.T @ misfit))

    def get_state(self):
        if isinstance(self.current_point, CUQIarray):
            self.current_point = self.current_point.to_numpy()
        if isinstance(self.current_target_eval, CUQIarray):
            self.current_target_eval = self.current_target_eval.to_numpy()
        if isinstance(self.current_target_grad_eval, CUQIarray):
            self.current_target_grad_eval = self.current_target_grad_eval.to_numpy()
        return {'sampler_type': 'MALA', 'current_point': self.current_point, \
                'current_target_eval': self.current_target_eval, \
                'current_target_grad_eval': self.current_target_grad_eval, \
                'scale': self.scale}

    def set_state(self, state):
        temp = CUQIarray(state['current_point'] , geometry=self.target.geometry)
        self.current_point = temp
        temp = CUQIarray(state['current_target_eval'] , geometry=self.target.geometry)
        self.current_target_eval = temp
        temp = CUQIarray(state['current_target_grad_eval'] , geometry=self.target.geometry)
        self.current_target_grad_eval = temp
        self.scale = state['scale']
