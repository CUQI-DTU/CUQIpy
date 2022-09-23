<div align="center">
<img src="https://cuqi-dtu.github.io/CUQIpy/_static/logo.png" alt="CUQIpy logo" width="250"/>
</div>

# Computational Uncertainty Quantification for Inverse Problems in python

[![pytest](https://github.com/CUQI-DTU/CUQIpy/actions/workflows/tests.yml/badge.svg)](https://github.com/CUQI-DTU/CUQIpy/actions/workflows/tests.yml)
[![docs](https://github.com/CUQI-DTU/CUQIpy/actions/workflows/docs.yml/badge.svg)](https://cuqi-dtu.github.io/CUQIpy/)

Computational Uncertainty Quantification for Inverse Problems in python (CUQIpy) is a python package for modeling and solving inverse problems in a Bayesian inference framework. CUQIpy provides a simple high-level interface to perform UQ analysis of inverse problems, while still allowing full control of the models and methods. The package comes equipped with a number of predefined distributions, samplers, models and test problems and is built to be easily further extended when needed.

You can find the full CUQIpy documentation [here](https://cuqi-dtu.github.io/CUQIpy/). 

This software package is part of the [CUQI project](https://www.compute.dtu.dk/english/cuqi) funded by [the Villum Foundation.](https://veluxfoundations.dk/en/forskning/teknisk-og-naturvidenskabelig-forskning)

## Quickstart
Install CUQIpy using pip:
```{r, engine='bash', count_lines}
pip install cuqipy
```
For more detailed instructions, see the [Getting Started](https://cuqi-dtu.github.io/CUQIpy/user/getting_started.html) guide.

## Quick Example - UQ in 5 steps
Image deconvolution with uncertainty quantification
```python
# Imports
import numpy as np
import matplotlib.pyplot as plt
from cuqi.testproblem import Deconvolution2D
from cuqi.data import grains
from cuqi.distribution import Laplace_diff, GaussianCov 
from cuqi.problem import BayesianProblem

# Step 1: Model and data, y = Ax
A, y_data, info = Deconvolution2D.get_components(dim=128, phantom=grains())

# Step 2: Prior, x ~ Laplace_diff(0, 0.01)
x = Laplace_diff(location=np.zeros(A.domain_dim),
                 scale=0.01,
                 bc_type='neumann',
                 physical_dim=2)

# Step 3: Likelihood, y ~ N(Ax, 0.0036^2)
y = GaussianCov(mean=A@x, cov=0.0036**2)

# Step 4: Set up Bayesian problem and sample posterior
BP = BayesianProblem(y, x).set_data(y=y_data)
samples = BP.sample_posterior(200)

# Step 5: Analysis
info.exactSolution.plot(); plt.title("Exact solution")
y_data.plot(); plt.title("Data")
samples.plot_mean(); plt.title("Posterior mean")
samples.plot_std(); plt.title("Posterior standard deviation")
```

<p float="left">
<img src="https://cuqi-dtu.github.io/CUQIpy/_images/deconv2D_exact_sol.png" alt="Exact solution" width="330">
<img src="https://cuqi-dtu.github.io/CUQIpy/_images/deconv2D_data.png" alt="Data" width="330">
<img src="https://cuqi-dtu.github.io/CUQIpy/_images/deconv2D_post_mean.png" alt="Posterior mean" width="330">
<img src="https://cuqi-dtu.github.io/CUQIpy/_images/deconv2D_post_std.png" alt="Posterior standard deviation" width="330">
</p>

## Plugins
CUQIpy can be extended with additional functionality by installing optional plugins. These can be found at
[CUQI-DTU](https://github.com/CUQI-DTU?q=CUQIpy-).

## Contributing
We welcome contributions to CUQIpy. Please see our [contributing guidelines](https://cuqi-dtu.github.io/CUQIpy/dev/index.html) for more information.

## Contributors

See the list of
[contributors](https://github.com/CUQI-DTU/CUQIpy/graphs/contributors)
who participated in this project.
