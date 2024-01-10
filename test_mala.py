# %%
import numpy as np
import cuqi
from cuqi.sampler import Sampler
import matplotlib.pyplot as plt
import pickle as pkl
from cuqi.array import CUQIarray
# %%
class MALA_new(cuqi.sampler.SamplerNew):
    def __init__(self, target, initial_point=None, scale=1.0, callback=None):
        super().__init__(target, initial_point, callback)
        self.scale = scale
        self.current_point = initial_point
        self.current_target_eval = self.target.logd(self.current_point)
        self.current_target_grad_eval = self.target.gradient(self.current_point)
        self._acc = [1]
        self.batch_size = 0
        self.num_batch_dumped = 0

    def step(self):
        xi = cuqi.distribution.Normal(mean=np.zeros(self.dim), std=np.sqrt(self.scale)).sample()
        x_star = self.current_point + 0.5*self.scale*self.current_target_grad_eval + xi

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
        return {'sampler_type': 'MALA', 'current_point': self.current_point.to_numpy(), \
                'current_target_eval': self.current_target_eval.to_numpy(), \
                'current_target_grad_eval': self.current_target_grad_eval.to_numpy(), \
                'scale': self.scale}

    def set_state(self, state):
        temp = CUQIarray(state['current_point'] , geometry=self.target.geometry)
        self.current_point = temp
        temp = CUQIarray(state['current_target_eval'] , geometry=self.target.geometry)
        self.current_target_eval = temp
        temp = CUQIarray(state['current_target_grad_eval'] , geometry=self.target.geometry)
        self.current_target_grad_eval = temp
        self.scale = state['scale']
#%%
# Define program
def make_custom_distribution(dim=5):
    mu = np.arange(dim)  # Mean of Gaussian
    std = 1  # standard deviation of Gaussian

    # Logpdf function
    logpdf_func = lambda x: -1/(std**2)*np.sum((x-mu)**2)
    gradient_func = lambda x: -2/(std**2)*(x-mu)

    # Define distribution from logpdf as UserDefinedDistribution (sample and gradients also supported)
    target = cuqi.distribution.UserDefinedDistribution(dim=dim, logpdf_func=logpdf_func,
                                                        gradient_func=gradient_func)
    return target

dim = 5
target = make_custom_distribution(dim)
eps = 1/dim
N = 2000
x0 = np.zeros(dim)

# %% Compare old MALA sampler vs new MALA sampler
# Set up old MALA sampler
np.random.seed(0)
sampler_old = cuqi.sampler.MALA(target, scale=eps**2, x0=x0)
# Sample
samples_old = sampler_old.sample(N)
plt.figure()
samples_old.plot_chain()
plt.title('old MALA')
# Set up new MALA sampler
np.random.seed(0)
sampler = MALA_new(target, scale=eps**2, initial_point=x0)
# Sample
sampler.sample(N)
samples = sampler.get_samples()

plt.figure()
plt.plot(samples.samples[:,0])
plt.plot(samples.samples[:,1])
plt.plot(samples.samples[:,2])
plt.plot(samples.samples[:,3])
plt.plot(samples.samples[:,4])
plt.title('new MALA')

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

sampler2 = MALA_new(target, scale=eps**2, initial_point=x0)

sampler2.load_checkpoint('checkpoint.pickle')

np.random.seed(0)

sampler2.sample(1000)
axes[1].plot(samples.samples[:,1])
axes[1].set_title('with loaded checkpoint')
plt.show()
