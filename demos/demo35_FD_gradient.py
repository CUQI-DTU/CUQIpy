# %% Import required packages
import cuqi
import matplotlib.pyplot as plt

#%% Create test problem
TP = cuqi.testproblem.Heat_1D(dim=60, max_time=0.01, field_type='Step')

#%% Get the posterior
posterior = TP.posterior

#%% Sample from the posterior using Metropolis-Hastings
MH_sampler = cuqi.sampler.MetropolisHastings(posterior)
MH_samples = MH_sampler.sample_adapt(1000)
MH_samples.plot_ci(95,exact=TP.exactSolution)

#%% Sample from the posterior using MALA
plt.figure()
posterior.enable_FD()
MALA_sampler = cuqi.sampler.MALA(posterior, 0.0001)
MALA_samples = MALA_sampler.sample_adapt(1000)
MALA_samples.plot_ci(95,exact=TP.exactSolution)

#%% Plot the ESS of the two chains
plt.figure()
plt.plot(MH_samples.compute_ess(), label='MH ESS', marker='o')
plt.plot(MALA_samples.compute_ess(), label='MALA ESS', marker='o')
plt.legend()
