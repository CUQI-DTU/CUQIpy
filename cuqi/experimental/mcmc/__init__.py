"""
Re-implementation of sampler module in a more object-oriented way.

Main changes for users
----------------------

1. Sampling API
   ^^^^^^^^^^^^

   Previously one would call the `.sample` or `sample_adapt` methods of a sampler instance at :py:mod:`cuqi.sampler` to sample from a target distribution and store the samples as the output as follows:

   .. code-block:: python

      from cuqi.sampler import MH
      from cuqi.distribution import DistributionGallery

      # Target distribution
      target = DistributionGallery("donut")

      # Set up sampler
      sampler = MH(target)

      # Sample from the target distribution (Alternatively calling sample with explicit scale parameter set in sampler)
      samples = sampler.sample_adapt(Ns=100, Nb=100) # Burn-in (Nb) removed by default

   This has now changed to to a more object-oriented API which provides more flexibility and control over the sampling process.

   For example one can now more explicitly control when the sampler is tuned (warmup) and when it is sampling with fixed parameters.

   .. code-block:: python

      from cuqi.experimental.mcmc import MH
      from cuqi.distribution import DistributionGallery

      # Target distribution
      target = DistributionGallery("donut")

      # Set up sampler
      sampler = MH(target)

      # Sample from the target distribution
      sampler.warmup(Nb=100)  # Explicit warmup (tuning) of sampler
      sampler.sample(Ns=100)  # Sampling with fixed parameters
      samples = sampler.get_samples().burnthin(Nb=100) # Getting samples and removing burn-in from warmup

   Importantly, the removal of burn-in from e.g. warmup is now a separate step that is done after the sampling process is complete.

2. Sampling API for BayesianProblem
   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   :py:class:`cuqi.problem.BayesianProblem` continues to have the same API for `sample_posterior` and the `UQ` method.

   There is now a flag `experimental` that can be set to `True` to use the new MCMC samplers.
   
   By default, the flag is set to `False` and the old samplers are used.

   For this more high-level interface, burn-in is automatically removed from the samples as was the case before.


3. More options for Gibbs sampling
   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   There are now more options for Gibbs sampling. Previously it was only possible to sample with Gibbs for samplers  :py:class:`cuqi.sampler.LinearRTO`, :py:class:`cuqi.sampler.RegularizedLinearRTO`, :py:class:`cuqi.sampler.Conjugate`, and :py:class:`cuqi.sampler.ConjugateApprox`.

   Now, it is possible to define a Gibbs sampling scheme using any sampler from the :py:mod:`cuqi.experimental.mcmc` module.

   **Example using a NUTS-within-Gibbs scheme for a 1D deconvolution problem:**

   .. code-block:: python

      import cuqi
      import numpy as np
      from cuqi.distribution import Gamma, Gaussian, GMRF, JointDistribution
      from cuqi.experimental.mcmc import NUTS, HybridGibbs, Conjugate
      from cuqi.testproblem import Deconvolution1D

      # Forward problem
      A, y_data, info = Deconvolution1D(dim=128, phantom='sinc', noise_std=0.001).get_components()

      # Bayesian Inverse Problem
      s = Gamma(1, 1e-4)
      x = GMRF(np.zeros(A.domain_dim), 50)
      y = Gaussian(A @ x, lambda s: 1 / s)

      # Posterior
      target = JointDistribution(y, x, s)(y=y_data)

      # Gibbs sampling strategy. Note we can define initial_points and various parameters for each sampler
      sampling_strategy = {
          "x": NUTS(max_depth=10, initial_point=np.zeros(A.domain_dim)),
          "s": Conjugate()
      }

      # Here we do 10 internal steps with NUTS for each Gibbs step
      num_sampling_steps = {
          "x": 10,
          "s": 1
      }

      sampler = HybridGibbs(target, sampling_strategy, num_sampling_steps)

      sampler.warmup(50)
      sampler.sample(200)
      samples = sampler.get_samples().burnthin(Nb=50)

      samples["x"].plot_ci(exact=info.exactSolution)
"""



from ._sampler import Sampler, ProposalBasedSampler
from ._langevin_algorithm import ULA, MALA, MYULA, PnPULA
from ._mh import MH
from ._pcn import PCN
from ._rto import LinearRTO, RegularizedLinearRTO
from ._cwmh import CWMH
from ._laplace_approximation import UGLA
from ._hmc import NUTS
from ._gibbs import HybridGibbs
from ._conjugate import Conjugate
from ._conjugate_approx import ConjugateApprox
from ._direct import Direct
from ._utilities import find_valid_samplers
