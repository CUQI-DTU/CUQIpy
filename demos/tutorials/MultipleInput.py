# %%
"""
Bathtub demo
==============

This is a demo for a bathtub temperature and volume model using CUQIpy. 
We have measurements of the temperature and volume of the water in the bathtub
and want to infer the temperature and volume of hot water and cold water that
were used to fill in the bathtub"""

# %%
# Import libraries
# ----------------

import cuqi
import numpy as np

# %%
# Define the forward map
# --------------------------
#
# `h_v` is the volume of hot water, `h_t` is the temperature of hot water,
# `c_v` is the volume of cold water, `c_t` is the temperature of cold water.

def forward_map(h_v, h_t, c_v, c_t): 
    # volume
    volume = h_v + c_v
    # temperature
    temp = (h_v * h_t + c_v * c_t) / (h_v + c_v)

    return np.array([volume, temp]).reshape(2,)

# %%
# Define the gradients
# --------------------------


# Define the gradient with respect to h_v
def gradient_h_v(direction, h_v, h_t, c_v, c_t):
    return (
        direction[0]
        + (h_t / (h_v + c_v) - (h_v * h_t + c_v * c_t) / (h_v + c_v) ** 2)
        * direction[1]
    )

# Define the gradient with respect to h_t
def gradient_h_t(direction, h_v, h_t, c_v, c_t):
    return (h_v / (h_v + c_v)) * direction[1]

# Define the gradient with respect to c_v
def gradient_c_v(direction, h_v, h_t, c_v, c_t):
    return (
        direction[0]
        + (c_t / (h_v + c_v) - (h_v * h_t + c_v * c_t) / (h_v + c_v) ** 2)
        * direction[1]
    )

# Define the gradient with respect to c_t
def gradient_c_t(direction, h_v, h_t, c_v, c_t):
    return (c_v / (h_v + c_v)) * direction[1]


# %%
# Define domain geometry and range geometry
# ---------------------------------------------

domain_geometry = (
    cuqi.geometry.Discrete(['h_v']),
    cuqi.geometry.Discrete(['h_t']),
    cuqi.geometry.Discrete(['c_v']),
    cuqi.geometry.Discrete(['c_t'])
)

range_geometry = cuqi.geometry.Discrete(['temperature','volume'])

# %%
# Define the forward model object
# --------------------------------------

model = cuqi.model.Model(
    forward=forward_map,
    gradient=(gradient_h_v, gradient_h_t, gradient_c_v, gradient_c_t),
    domain_geometry=domain_geometry,
    range_geometry=range_geometry
)

# %%
# Experiment with partial evaluation of the model
# ---------------------------------------------------

print("\nmodel()\n", model())
print("\nmodel(h_v = 50)\n", model(h_v=50))
print("\nmodel(h_v = 50, h_t = 60)\n", model(h_v=50, h_t=60))
print("\nmodel(h_v = 50, h_t = 60, c_v = 30)\n", model(h_v=50, h_t=60, c_v=30))
print(
    "\nmodel(h_v = 50, h_t = 60, c_v = 30, c_t = 10)\n",
    model(h_v=50, h_t=60, c_v=30, c_t=10),
)

# %%
# Define prior distributions for the unknown parameters
# ------------------------------------------------------------

h_v_dist = cuqi.distribution.Uniform(0, 60, geometry=domain_geometry[0])
h_t_dist = cuqi.distribution.Uniform(40, 70, geometry=domain_geometry[1])
c_v_dist = cuqi.distribution.Uniform(0, 60, geometry=domain_geometry[2])
c_t_dist = cuqi.distribution.TruncatedNormal(
    10, 2**2, 7, 15, geometry=domain_geometry[3]
)

# %%
# Define a data distribution
# ----------------------------

data_dist = cuqi.distribution.Gaussian(
    mean=model(h_v_dist, h_t_dist, c_v_dist, c_t_dist),
    cov=np.array([[1**2, 0], [0, 0.5**2]])
)

# %%
# Define a joint distribution of prior and data distributions
# ---------------------------------------------------------------

joint_dist = cuqi.distribution.JointDistribution(
    data_dist,
    h_v_dist,
    h_t_dist,
    c_v_dist,
    c_t_dist
)

# %%
# Define the posterior distribution by setting the observed data
# ------------------------------------------------------------------

posterior = joint_dist(data_dist=np.array([60, 38]))

# %%
# Experiment with conditioning the posterior distribution
# ------------------------------------------------------------

print("posterior", posterior)
print("\nposterior(h_v = 50)\n", posterior(h_v_dist=50))
print("\nposterior(h_v = 50, h_t = 60)\n", posterior(h_v_dist=50, h_t_dist=60))
print(
    "\nposterior(h_v = 50, h_t = 60, c_v = 30)\n",
    posterior(h_v_dist=50, h_t_dist=60, c_v_dist=30),
)

# %%
# Sample from the joint distribution
# ------------------------------------------------------------
#
# First define sampling strategy for Gibbs sampling

sampling_strategy = {
    "h_v_dist": cuqi.experimental.mcmc.MALA(
        scale=0.2, initial_point=np.array([30])),
    "h_t_dist": cuqi.experimental.mcmc.MALA(
        scale=0.2, initial_point=np.array([50])),
    "c_v_dist": cuqi.experimental.mcmc.MALA(
        scale=0.2, initial_point=np.array([30])),
    "c_t_dist": cuqi.experimental.mcmc.MALA(
        scale=0.2, initial_point=np.array([10])),
}

# %%
# Then create the sampler and sample

hybridGibbs = cuqi.experimental.mcmc.HybridGibbs(
    posterior,
    sampling_strategy=sampling_strategy)

hybridGibbs.warmup(100)
hybridGibbs.sample(2000)
samples = hybridGibbs.get_samples()

# %%
# Plot some results
# ------------------

# Compute mean values
mean_h_v = samples['h_v_dist'].mean()
mean_h_t = samples['h_t_dist'].mean()
mean_c_v = samples['c_v_dist'].mean()
mean_c_t = samples['c_t_dist'].mean()

# Print mean values
print(f"Mean h_v: {mean_h_v}, Mean h_t: {mean_h_t}, Mean c_v: {mean_c_v}, Mean c_t: {mean_c_t}")
print("Measured volume:", 60)
print("Mean predicted volume:", mean_h_v + mean_c_v)
print()
print("Measured temperature:", 38)
print("Mean predicted temperature:", (mean_h_v * mean_h_t + mean_c_v * mean_c_t) / (mean_h_v + mean_c_v))

# Plot trace of samples
samples['h_v_dist'].plot_trace()
samples['h_t_dist'].plot_trace()
samples['c_v_dist'].plot_trace()
samples['c_t_dist'].plot_trace()
