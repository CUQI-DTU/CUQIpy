# %%
import cuqi
import matplotlib.pyplot as plt

#%%
TP = cuqi.testproblem.Heat_1D(dim=60, max_time=0.01, field_type='Step')

posterior = TP.posterior

MH_sampler = cuqi.sampler.MetropolisHastings(posterior)
MH_samples = MH_sampler.sample_adapt(1000)
MH_samples.plot_ci(95,exact=TP.exactSolution)

#%%
plt.figure()
posterior.use_FD = True
MALA_sampler = cuqi.sampler.MALA(posterior, 0.0001)
MALA_samples = MALA_sampler.sample_adapt(1000)
MALA_samples.plot_ci(95,exact=TP.exactSolution)

#%%
plt.figure()
plt.plot(MH_samples.compute_ess(), label='MH ESS', marker='o')
plt.plot(MALA_samples.compute_ess(), label='MALA ESS', marker='o')
plt.legend()
