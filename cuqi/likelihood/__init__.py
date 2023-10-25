""" 
The :class:`~cuqi.likelihood.Likelihood` represents the likelihood function commonly used in conjunction
with a prior to define a :class:`cuqi.distribution.Posterior` or :class:`cuqi.problem.BayesianProblem`.

Quick example
-------------
Create a Gaussian likelihood function from a forward `model` and observed `data`

.. code-block:: python

   import cuqi
   model, data, probInfo = cuqi.testproblem.Deconvolution1D().get_components()
   likelihood = cuqi.distribution.Gaussian(mean=model, cov=0.05**2).to_likelihood(data)


Mathematical details
--------------------

Given a conditional distribution :math:`\pi(b \mid x)` and a observed data :math:`b^{obs}` the likelihood function is defined as

.. math::
   
   L(x \mid b^{obs}): x \to \pi(b^{obs} \mid x).

The most commonly use-case of the likelihood function is to define the likelihood function for a Bayesian inverse problem with Gaussian measurement noise.

Consider the inverse problem

.. math::

   b^{obs} = A(x)+e,

where :math:`b^{obs}\in\mathbb{R}^m` is the (noisy) measurement, :math:`e\in\mathbb{R}^m` is the measurement noise, :math:`x\in\mathbb{R}^n` is the solution to the inverse problem and :math:`A: \mathbb{R}^n \\to \mathbb{R}^m` is the forward model.

The stochastic extension of the inverse problem is

.. math::
   B = AX+E,

where :math:`B, X` and :math:`E` are random variables.

Assuming Gaussian measurement noise :math:`E\sim\mathcal{N}(0, \sigma^2 I)` the data follows the distribution :math:`B \mid X=x \sim\mathcal{N}(A(x),\sigma^2I)` and given an observation :math:`b^{obs}` the likelihood function is given by

.. math::

   L(x \mid b^{obs}) \propto \exp\left( -\\frac{1}{2\sigma^2} \|b^{obs}-Ax\|_2^2 \\right).

"""

from ._likelihood import Likelihood, UserDefinedLikelihood
