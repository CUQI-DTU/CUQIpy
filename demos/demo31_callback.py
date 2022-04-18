# %%
import sys; sys.path.append("..")
import cuqi
import numpy as np
import matplotlib.pyplot as plt

#Interactive plot in popup window (tk, qt4, qt5, etc.)
%matplotlib qt 

# %% Define testproblem
TP = cuqi.testproblem.Deconvolution2D(phantom=cuqi.data.grains())
TP.prior = cuqi.distribution.Laplace_diff(location=np.zeros(TP.model.domain_dim),
                                          scale=0.01,
                                          bc_type="neumann",
                                          physical_dim=2,
                                          geometry=TP.model.domain_geometry)

# %% Sample while also plotting progress

# Define callback function with structure (sample, n)
# Prepare figure
fig, ax = plt.subplots()
fig.canvas.draw()
fig.canvas.flush_events()
def callback(sample, n):

    if n % 10 == 0: #Only plot every 50th sample
        # Clear axis
        ax.cla() 

        # Plot samples
        ax.imshow(sample.reshape(TP.model.domain_geometry.shape), cmap="gray")
        ax.set_title(f'Sample {n}')

        # Update figure
        fig.canvas.draw() 
        fig.canvas.flush_events() #ensure

# Run sampler with callback plotting
samples = TP.sample_posterior(500, callback=callback)

# %%
