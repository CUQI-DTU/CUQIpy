from cuqi.distribution import JointDistribution
from cuqi.sampler import Sampler, Gibbs
from cuqi.samples import Samples
from typing import Dict, Union
import numpy as np
import sys
import matplotlib.pyplot as plt

import cuqi
d = cuqi.distribution.Gamma(1, 1)
x = cuqi.distribution.Gaussian(0, lambda d: 1/d)
# d = cuqi.distribution.Gaussian(-3, 1)
# x = cuqi.distribution.Gaussian(3, 1)

class Direct:
    def __init__(self, target):
        self.target = target
    
    def step(self, x=None):
        """ Take a single step of the sampler. """
        return self.target.sample()
        # return x

class MYMH:
    def __init__(self, target):
        self.target = target
        self.sampler = cuqi.sampler.MH(target, x0=1, scale=0.1)
    def step(self, x=None):
        self.sampler.x0 = x
        temp = self.sampler.sample(10)
        temp = temp.samples
        print(temp)
        result = temp[:,-1].reshape(x.shape)
        print(temp[:,-1].reshape(x.shape))
        print(x)
        return result

class MYMHNEW:
    def __init__(self, target):
        self.target = target
        self.sampler = cuqi.sampler.MH_new(target, x0=1, scale=0.1)
    def step(self, x=None):
        # TODO: find a better way to set the state
        state = {'sampler_type': 'MH', 'current_point': x, 'current_target': self.target.logd(x), 'scale': self.sampler.scale}
        self.sampler.set_state(state)
        temp = self.sampler.sample(9)
        print(temp)
        result = self.sampler.current_point
        print(x)
        return result

target = cuqi.distribution.JointDistribution(x, d)

# Define the sampling strategy
sampling_strategy = {
    'x' : MYMH,
    'd' : MYMH,
}
sampling_strategy_new = {
    'x' : MYMHNEW,
    'd' : MYMHNEW,
}
np.random.seed(0)

sampler = Gibbs(target, sampling_strategy)

samples = sampler.sample(100)

plt.figure()
plt.plot(samples['d'].samples[0])
plt.plot(samples['x'].samples[0])
plt.title("Gibbs with old MH")

np.random.seed(0)

sampler = Gibbs(target, sampling_strategy_new)

samples = sampler.sample(100)

plt.figure()
plt.plot(samples['d'].samples[0])
plt.plot(samples['x'].samples[0])
plt.title("Gibbs with new MH")