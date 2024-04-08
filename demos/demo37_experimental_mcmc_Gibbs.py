# %%
import numpy as np
from cuqi.distribution import Gamma, Gaussian, GMRF, JointDistribution
from cuqi.experimental.mcmc import MHNew, NUTSNew, GibbsNew, ConjugateNew
from cuqi.sampler import Gibbs, Conjugate, LinearRTO, NUTS
from cuqi.testproblem import Deconvolution1D

# %%
A, y_data, info = Deconvolution1D(dim=64, phantom='sinc', noise_std=0.001).get_components()

# %%
s = Gamma(1, 1e-4)
x = GMRF(np.zeros(A.domain_dim), 50)
y = Gaussian(A@x, lambda s: 1/s)

# %%
target = JointDistribution(y, x, s)(y=y_data)

# %%
# Now using NUTS

sampling_strategy2 = {
    "x" : NUTSNew(),
    "s" : ConjugateNew()
}

sampler2 = GibbsNew(target, sampling_strategy2)

# %%
samples2 = sampler2.sample(50, 40)

# %%
samples2["x"].plot_ci(exact=info.exactSolution)

# %%
samples2["s"].plot_trace(exact=0.01**2)

# %%
target_cond = target(s=1/0.001**2)

# %%
# Sample with old NUTS
sampler_old = NUTS(target_cond)
samples_old = sampler_old.sample(200, 50)
samples_old.plot_ci(exact=info.exactSolution)


# %%
# Sample with new NUTS
sampler3 = NUTSNew(target_cond)
sampler3.warmup(200)
sampler3.sample(200)
samples3 = sampler3.get_samples()
samples3.plot_ci(exact=info.exactSolution)


# %%

sampling_strategy = {
    "x" : LinearRTO,
    "s" : Conjugate
}

sampler = Gibbs(target, sampling_strategy)

# %%

samples = sampler.sample(200, 50)

# %%
samples["x"].plot_ci(exact=info.exactSolution)
# %%
samples["s"].plot_trace(exact=0.01**2)



# %%

# %%
