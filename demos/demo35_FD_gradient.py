# %% Import required packages
import cuqi
import matplotlib.pyplot as plt

#%% Create test problem
TP = cuqi.testproblem.Heat1D(dim=60, max_time=0.01, field_type='Step')

#%% Get the posterior
posterior = TP.posterior

#%% Sample from the posterior using Metropolis-Hastings
print('Sampling from the posterior using Metropolis-Hastings:')
MH_sampler = cuqi.sampler.MH(posterior)
MH_samples = MH_sampler.sample_adapt(1000, 100)
plt.figure()
MH_samples.plot_ci(95,exact=TP.exactSolution)
plt.title("MH")

#%% Sample from the posterior using MALA
print('Sampling from the posterior using MALA:')
print('Attempt sampling:')
try:
    MALA_sampler = cuqi.sampler.MALA(posterior, 0.0001)
    MALA_samples = MALA_sampler.sample_adapt(1000)
except Exception as e:
    print(e)
print('Sampling failed because the gradient of the posterior is not available.')

#%%
print('Enable finite difference approximation of the gradient ' +
      'for the posterior, and attempt sampling again using MALA, ULA and NUTS:')

posterior.enable_FD()
MALA_sampler = cuqi.sampler.MALA(posterior, 0.0001)
MALA_samples = MALA_sampler.sample_adapt(1000, 10)
plt.figure()
MALA_samples.plot_ci(95,exact=TP.exactSolution)
plt.title("MALA")

#%% Sample from the posterior using ULA
ULA_sampler = cuqi.sampler.ULA(posterior, 0.0001)
ULA_samples = ULA_sampler.sample_adapt(1000, 10)
plt.figure()
ULA_samples.plot_ci(95,exact=TP.exactSolution)
plt.title("ULA")

#%% Sample from the posterior using NUTS
NUTS_sampler = cuqi.sampler.NUTS(posterior)
NUTS_samples = NUTS_sampler.sample_adapt(1000, 10)
plt.figure()
NUTS_samples.plot_ci(95, exact=TP.exactSolution)
plt.title("NUTS")

#%% Plot the ESS of all the chains
plt.figure()
plt.plot(MH_samples.compute_ess(), label='MH ESS', marker='o')
plt.plot(MALA_samples.compute_ess(), label='MALA ESS', marker='o')
plt.plot(ULA_samples.compute_ess(), label='ULA ESS', marker='o')
plt.plot(NUTS_samples.compute_ess(), label='NUTS ESS', marker='o')
plt.legend()