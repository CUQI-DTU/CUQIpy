API reference
=============

CUQIpy is a large library of sub modules. The following is an overview of the overall structure of the library.

The main modules are:
   - :doc:`cuqi.distribution <_autosummary/cuqi.distribution>` for defining probability distributions.
   - :doc:`cuqi.model <_autosummary/cuqi.model>` for defining deterministic models.
   - :doc:`cuqi.sampler <_autosummary/cuqi.sampler>` for sampling from probability distributions.

The following modules provide higher-level interfaces:
   - :doc:`cuqi.testproblem <_autosummary/cuqi.testproblem>` for defining specific test problems.
   - :doc:`cuqi.problem <_autosummary/cuqi.problem>` for automatic UQ by simply defining problem.
   - :doc:`cuqi.pde <_autosummary/cuqi.pde>` for defining and solving PDEs.

Some useful auxiliary helper modules:
   - :doc:`cuqi.likelihood <_autosummary/cuqi.likelihood>` for defining likelihood functions.
   - :doc:`cuqi.geometry <_autosummary/cuqi.geometry>` for defining the geometry of objects like distributions or models.
   - :doc:`cuqi.samples <_autosummary/cuqi.samples>` contains tools for storing and manipulating MCMC samples.
   - :doc:`cuqi.solver <_autosummary/cuqi.solver>` contains tools point estimation of posteriors.

Full API overview
   A complete auto generated overview of the API can be found below (or by navigating using the sidebar).

.. autosummary::
   :toctree: _autosummary
   :nosignatures:
   :recursive:

   cuqi