# %%
import sys; sys.path.append("..")
import cuqi
import numpy as np
import matplotlib.pyplot as plt

#Interactive plot in popup window (tk, qt4, qt5, etc.)
try:
    get_ipython().run_line_magic('matplotlib', 'qt')
except Exception:
    pass


# %% Use a callback function to plot the current sample every n'th sample

# Define testproblem
TP = cuqi.testproblem.Deconvolution2D(phantom=cuqi.data.grains())

# Define prior
TP.prior = cuqi.distribution.LMRF(
    location=0,
    scale=0.01,
    bc_type="neumann",
    geometry=TP.model.domain_geometry
)

# Prepare figure
fig, ax = plt.subplots()
fig.canvas.draw()
fig.canvas.flush_events()

# Define callback function with NEW signature
def callback(sampler, sample_index, num_of_samples):

    if sample_index % 10 == 0: #Only plot every n'th sample
        # Clear axis
        ax.cla() 

        # Get current sample from sampler
        sample = sampler.current_point
        
        # Plot samples
        ax.imshow(sample.reshape(TP.model.domain_geometry.fun_shape), cmap="gray")
        ax.set_title(f'Sample index {sample_index}')

        # Update figure
        fig.canvas.draw() 
        fig.canvas.flush_events()

# Run sampler with callback plotting
samples = TP.sample_posterior(500, callback=callback)

# %% Example of using callback to store all samples and plot statistics

def make_callback_function(fig, TP, Ns):
    """ Method to create a callback function that gets called by CUQIpy for each sample

    Parameters
    ----------
    fig : matplotlib figure
        Figure to plot progress in

    TP : cuqi.testproblem.TestProblem
        Test problem that is sampled from (used to get parameters)

    Ns : int
        Number of samples to draw

    Returns
    -------
    callback : function
        Callback function that we feed to CUQIpy
    
    Notes
    -----
    The current implementation stores the samples in a NumPy array and periodically creates a CUQI samples object and plots the CI.

    """

    # Preallocate samples array to compute statistics along the way
    # TODO: In future versions of CUQIpy we can access the samples object directly in the callback function
    samples = np.zeros((TP.model.domain_dim, Ns))

    # Create callback function with signature (sampler, sample_index, num_of_samples)
    def callback(sampler, sample_index, num_of_samples):

        if sample_index % 10 == 0:
            # Clear axis
            ax.cla() 

            # Get current sample from sampler
            sample = sampler.current_point
            
            # Plot samples - use plot() for 1D data
            ax.plot(sample)
            ax.set_title(f'Sample index {sample_index}')

            # Update figure
            fig.canvas.draw() 
            fig.canvas.flush_events()

    return callback

# Prepare figure
fig, ax = plt.subplots()
fig.canvas.draw() 
fig.canvas.flush_events()

# Number of samples to draw
Ns = 500

# Define test problem and prior
TP = cuqi.testproblem.Deconvolution1D(phantom="square") # Default values
TP.prior = cuqi.distribution.LMRF(0, 0.01, geometry=TP.model.domain_geometry) # Set prior

# Create callback function for progress plotting (Burn-in is 20% by default so we allocate 120% of samples)
callback = make_callback_function(fig, TP, int(1.2*Ns))

# Sample posterior
xs = TP.sample_posterior(Ns, callback=callback)
