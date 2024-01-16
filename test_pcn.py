# %%
import numpy as np
import cuqi
from cuqi.sampler import Sampler
import matplotlib.pyplot as plt
import pickle as pkl
from cuqi.array import CUQIarray
# %%
class PCN_new(cuqi.sampler.SamplerNew):
    def __init__(self, target, initial_point=None, scale=1.0, callback=None):
        super().__init__(target, initial_point, callback)
        self.scale = scale
        self.current_point = initial_point

        self.current_loglike_eval = self._loglikelihood(self.current_point)

        self._acc = [1]
        self.batch_size = 0
        self.num_batch_dumped = 0

    def step(self):
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

    @Sampler.target.setter 
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
    def dim(self):
        if hasattr(self,'target') and hasattr(self.target,'dim'):
            self._dim = self.target.dim
        elif hasattr(self,'target') and isinstance(self.target,tuple) and len(self.target)==2:
            self._dim = self.target[0].dim
        return self._dim

    def tune(self, skip_len, update_count):
        pass

    def log_proposal(self, theta_star, theta_k, g_logpi_k):
        mu = theta_k + ((self.scale)/2)*g_logpi_k
        misfit = theta_star - mu
        return -0.5*((1/(self.scale))*(misfit.T @ misfit))

    def current_point(self):
        return self.current_point

    def dump_samples(self):
        np.savez( self.sample_path + 'batch_{:04d}.npz'.format( self.num_batch_dumped), samples=np.array(self._samples[-1-self.batch_size:] ), batch_id=self.num_batch_dumped )
        self.num_batch_dumped += 1

    def save_checkpoint(self, path):
        state = self.get_state()

        with open(path, 'wb') as handle:
            pkl.dump(state, handle, protocol=pkl.HIGHEST_PROTOCOL)

    def load_checkpoint(self, path):
        with open(path, 'rb') as handle:
            state = pkl.load(handle)

        self.set_state(state)

    def reset(self):
        self._samples.clear()
        self._acc.clear()

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
#%%
# Define custom distribution
# this is from tests/test_pCN_sample_regression.py
def make_custom_distribution(d):
    np.random.seed(0)
    mu = np.zeros(d)
    sigma = np.linspace(0.5, 1, d)
    model = cuqi.model.Model(lambda x: x, range_geometry=d, domain_geometry=d)
    L = cuqi.distribution.Gaussian(mean=model, sqrtcov=sigma).to_likelihood(np.zeros(d))
    def target(x): return L.logd(x)
    P = cuqi.distribution.Gaussian(mu, np.ones(d))
    target = cuqi.distribution.Posterior(L, P)
    return target

dim = 2
target = make_custom_distribution(dim)
scale = 0.1
x0 = 0.5*np.ones(dim)

np.random.seed(0)

MCMC = cuqi.sampler.pCN(target, scale, x0)
results = MCMC.sample(1000)
results.plot_chain()
plt.title("old PCN")
# %%
np.random.seed(0)
sampler = PCN_new(target, scale=scale, initial_point=x0)
# Sample
# TODO: there seems a bug with warmup, but I can't reproduce it
sampler.sample(1000)
samples = sampler.get_samples()
plt.figure()
plt.plot(samples.samples[:,0])
plt.plot(samples.samples[:,1])
plt.title("new PCN")

# %% test checkpointing with the new MALA sampler
np.random.seed(0)
sampler.sample(10000)

sampler.save_checkpoint('checkpoint.pickle')

np.random.seed(0)

sampler.reset()
sampler.sample(1000)

samples = sampler.get_samples()

f, axes = plt.subplots(1,2)

axes[0].plot(samples.samples[:,1])
axes[0].set_title('without checkpoint')

sampler2 = PCN_new(target, scale=0.1, initial_point=x0)

sampler2.load_checkpoint('checkpoint.pickle')

np.random.seed(0)

sampler2.sample(1000)
axes[1].plot(samples.samples[:,1])
axes[1].set_title('with loaded checkpoint')
plt.show()
