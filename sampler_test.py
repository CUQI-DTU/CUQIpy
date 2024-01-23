import sys
sys.path.append('./cuqi')

import numpy as np
import matplotlib.pyplot as plt
from cuqi.sampler import MH_new
from cuqi.distribution import Gaussian, JointDistribution

x = Gaussian(np.zeros(2), np.ones(2))
x0 = 2*np.ones(2)


#forward = lambda x: x
#y = Gaussian( forward, 0.01 * np.ones(2) )

#joint = JointDistribution(x,y)
#posterior = joint(y=np.array([0.1,0.2]) )

sampler = MH_new(x, x0=x0 , scale=0.1)

sampler.sample(10000)
sampler.warmup(10000)

sampler.save_checkpoint('checkpoint.pickle')

np.random.seed(0)

sampler.reset()



# user interface for doing sampling
# 1) current way
sam = sampler.sample(1000, warm_up=1000)
# potentialy reset sampler after sampling call (don't have to)
sam2 = sampler.sample(1000) # sample with the same state as the previous call
# 2) object oriented way
sampler.warmup(100)
sampler.sample(1000)
samples = sampler.get_samples()
# 3)
sam = sample("MH", x, x0=x0, scale=0.1, n_samples=1000, warm_up=1000)
# sampler is created and deleted dynamically
# 4)
# use option 2 for sampling under BayesianProblem's interface

# user interface for BayesianProblem
# 1)
BP.MAP()
# 2)
BP.MAP
# what happen we call it multiple times?
# 3)
BP.compute_MAP()
BP.get_MAP()















f, axes = plt.subplots(1,2)

axes[0].plot(samples.samples[:,1])

sampler2 = MH_new(x, x0=x0)

sampler2.load_checkpoint('checkpoint.pickle')

np.random.seed(0)

sampler2.sample(1000)
axes[1].plot(samples.samples[:,1])
plt.show()
