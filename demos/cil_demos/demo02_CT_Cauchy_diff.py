# %%
import sys
sys.path.append("../../") 
import numpy as np
from cuqi.distribution import Cauchy_diff
from cuqi.cil.testproblem import ParallelBeam2DProblem

# Computed Tomography
TP = ParallelBeam2DProblem(
    im_size=(256, 256),
    det_count=256,
    angles=np.linspace(0, np.pi, 180),
    phantom="shepp-logan",
)

# Cauchy difference prior
TP.prior = Cauchy_diff(
    location=np.zeros(TP.model.domain_dim),
    scale=0.01,
    physical_dim=2,
)

# Sample posterior with automatic sampler choice
samples = TP.sample_posterior(200)

# Plot sample mean and ci
samples.plot_ci()
