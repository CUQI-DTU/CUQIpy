
#%% Bathtub demo

import cuqi
import numpy as np

# Define the forward model
def forward_map(h_v, h_t, c_v, c_t): 
    # volume
    volume = h_v + c_v
    # temprature
    temp = (h_v * h_t + c_v * c_t) / (h_v + c_v)

    return np.array([volume, temp]).reshape(2,)

# TODO: fix the gradients and try MALA or NUTS
# Define the gradients
def gradient_h_v(direction, h_v, h_t, c_v, c_t):
    return direction[0] * 1 + h_t * direction[1]
#
#
def gradient_h_t(direction, h_v, h_t, c_v, c_t):
    return direction[0] * 0 + h_v * direction[1]
#
def gradient_c_v(direction, h_v, h_t, c_v, c_t):
    return direction[0] * 1 + c_t * direction[1]
#
def gradient_c_t(direction, h_v, h_t, c_v, c_t):
    return direction[0] * 0 + c_v * direction[1]

# Define the forward model with gradients
domain_geometry = (
    cuqi.geometry.Discrete(['h_v']),
    cuqi.geometry.Discrete(['h_t']),
    cuqi.geometry.Discrete(['c_v']),
    cuqi.geometry.Discrete(['c_t'])
)

range_geometry = cuqi.geometry.Continuous1D(2)#(['temperature','volume'])
model = cuqi.model.Model(
    forward=forward_map,
    gradient=(gradient_h_v, gradient_h_t, gradient_c_v, gradient_c_t),
    domain_geometry=domain_geometry,
    range_geometry=range_geometry
)

h_v = cuqi.distribution.Uniform(0, 60, geometry=domain_geometry[0])
h_t = cuqi.distribution.Uniform(40, 70, geometry=domain_geometry[1])
c_v = cuqi.distribution.Uniform(0, 60, geometry= domain_geometry[2])
c_t = cuqi.distribution.TruncatedNormal(10, 2**2, 7, 15, geometry=domain_geometry[3])  # Truncated normal distribution between 7 and 15

# Data distribution and likelihood
data_dist = cuqi.distribution.Gaussian(
    mean=model(h_v, h_t, c_v, c_t),
    cov=np.array([[1**2, 0], [0, 0.5**2]])
)


# Define the joint distribution
joint_dist = cuqi.distribution.JointDistribution(
    data_dist,
    h_v,
    h_t,
    c_v,
    c_t
)

posterior = joint_dist(data_dist=np.array([60, 38]))

# Sample from the joint distribution
sampling_strategy = {
    'h_v': cuqi.experimental.mcmc.MH(scale=0.05, initial_point= np.array([30])),
    'h_t': cuqi.experimental.mcmc.MH(scale=0.02, initial_point= np.array([50])),
    'c_v': cuqi.experimental.mcmc.MH(scale=0.05, initial_point= np.array([30])),
    'c_t': cuqi.experimental.mcmc.MH(scale=0.02, initial_point= np.array([10])),
}

#%% Sampler
hybridGibbs = cuqi.experimental.mcmc.HybridGibbs(
    posterior,
    sampling_strategy=sampling_strategy)

hybridGibbs.warmup(1000)
hybridGibbs.sample(2000)
samples = hybridGibbs.get_samples()

# Plot the results


# %%
mean_h_v = samples['h_v'].mean()
mean_h_t = samples['h_t'].mean()
mean_c_v = samples['c_v'].mean()
mean_c_t = samples['c_t'].mean()

print(f"Mean h_v: {mean_h_v}, Mean h_t: {mean_h_t}, Mean c_v: {mean_c_v}, Mean c_t: {mean_c_t}")

print("Measured volume:", 60)
print("Mean predicted volume:", mean_h_v + mean_c_v)
print()
print("Measured temperature:", 38)
print("Mean predicted temperature:", (mean_h_v * mean_h_t + mean_c_v * mean_c_t) / (mean_h_v + mean_c_v))

samples['h_v'].plot_trace()
samples['h_t'].plot_trace()
samples['c_v'].plot_trace()
samples['c_t'].plot_trace()
# %%
