<div align="center">
<img src="https://cuqi-dtu.github.io/CUQIpy/_static/logo.png" alt="CUQIpy logo" width="250"/>
</div>

# Computational Uncertainty Quantification for Inverse Problems in python

[![pytest](https://github.com/CUQI-DTU/CUQIpy/actions/workflows/tests.yml/badge.svg)](https://github.com/CUQI-DTU/CUQIpy/actions/workflows/tests.yml)
[![docs](https://github.com/CUQI-DTU/CUQIpy/actions/workflows/docs.yml/badge.svg)](https://cuqi-dtu.github.io/CUQIpy/)
[![PyPI](https://github.com/CUQI-DTU/CUQIpy/actions/workflows/publish_pypi.yml/badge.svg)](https://pypi.org/project/CUQIpy/#history)

**CUQIpy** stands for Computational Uncertainty Quantification for Inverse Problems in python. It's a robust Python package designed for modeling and solving inverse problems using Bayesian inference. Here's what it brings to the table:

- A straightforward high-level interface for UQ analysis.
- Complete control over the models and methods.
- An array of predefined distributions, samplers, models, and test problems.
- Easy extendability for your unique needs.

CUQIpy is part of the [CUQI project](https://www.compute.dtu.dk/english/cuqi) supported by the [Villum Foundation.](https://veluxfoundations.dk/en/forskning/teknisk-og-naturvidenskabelig-forskning)

## 📚 Resources

- **Documentation:** [CUQIpy website](https://cuqi-dtu.github.io/CUQIpy/)

- **Papers:** CUQIpy on ArXiv - [Part I](https://arxiv.org/abs/2305.16949) & [Part II](https://arxiv.org/abs/2305.16951)

- **CUQI book:** [CUQI book website](https://cuqi-dtu.github.io/CUQI-Book/)

- **User showcases:** [Showcase repository](https://github.com/CUQI-DTU/CUQIpy-User-Showcase/)

## 🚀 Quickstart
Install CUQIpy using pip:
```{r, engine='bash', count_lines}
pip install cuqipy
```
For more detailed instructions, see the [Getting Started](https://cuqi-dtu.github.io/CUQIpy/user/getting_started.html) guide.

## 🧪 Quick Example - UQ in a few lines of code
Experience the simplicity and power of CUQIpy with this Image deconvolution example. Getting started with UQ takes just a few lines of code:
```python
# Imports
import matplotlib.pyplot as plt
from cuqi.testproblem import Deconvolution2D
from cuqi.distribution import Gaussian, LMRF, Gamma
from cuqi.problem import BayesianProblem

# Step 1: Set up forward model and data, y = Ax
A, y_data, info = Deconvolution2D(dim=256, phantom="cookie").get_components()

# Step 2: Define distributions for parameters
d = Gamma(1, 1e-4)
s = Gamma(1, 1e-4)
x = LMRF(0, lambda d: 1/d, geometry=A.domain_geometry)
y = Gaussian(A@x, lambda s: 1/s)

# Step 3: Combine into Bayesian Problem and sample posterior
BP = BayesianProblem(y, x, d, s)
BP.set_data(y=y_data)
samples = BP.sample_posterior(200)

# Step 4: Analyze results
info.exactSolution.plot(); plt.title("Sharp image (exact solution)")
y_data.plot(); plt.title("Blurred and noisy image (data)")
samples["x"].plot_mean(); plt.title("Estimated image (posterior mean)")
samples["x"].plot_std(); plt.title("Uncertainty (posterior standard deviation)")
samples["s"].plot_trace(); plt.suptitle("Noise level (posterior trace)")
samples["d"].plot_trace(); plt.suptitle("Regularization parameter (posterior trace)")
```

<p float="left">
<img src="https://cuqi-dtu.github.io/CUQIpy/_images/deconv2D_exact_sol.png" alt="Sharp image (exact solution)" width="330">
<img src="https://cuqi-dtu.github.io/CUQIpy/_images/deconv2D_data.png" alt="Blurred and noisy image (data)" width="330">
<img src="https://cuqi-dtu.github.io/CUQIpy/_images/deconv2D_post_mean.png" alt="Estimated image (posterior mean)" width="330">
<img src="https://cuqi-dtu.github.io/CUQIpy/_images/deconv2D_post_std.png" alt="Uncertainty (posterior standard deviation)" width="330">
<img src="https://cuqi-dtu.github.io/CUQIpy/_images/deconv2D_noise_level.png" alt="Noise level (posterior trace)" width="660">
<img src="https://cuqi-dtu.github.io/CUQIpy/_images/deconv2D_regularization_parameter.png" alt="Regularization parameter (posterior trace)" width="660">
</p>

## 🔌 Plugins
CUQIpy can be extended with additional functionality by installing optional plugins. We currently offer the following plugins:

- [CUQIpy-CIL](https://github.com/CUQI-DTU/CUQIpy-CIL) A plugin for the Core Imaging Library [(CIL)](https://ccpi.ac.uk/cil/) providing access to forward models for X-ray computed tomography.

- [CUQIpy-FEniCS](https://github.com/CUQI-DTU/CUQIpy-FEniCS): A plugin providing access to the finite element modelling tool [FEniCS](https://fenicsproject.org), which is used for solving PDE-based inverse problems.

- [CUQIpy-PyTorch](https://github.com/CUQI-DTU/CUQIpy-PyTorch): A plugin providing access to the automatic differentiation framework of [PyTorch](https://pytorch.org) within CUQIpy. It allows gradient-based sampling methods without manually providing derivative information of distributions and forward models.

## 💻 Maintainers
- [Nicolai André Brogaard Riis](https://github.com/nabriis)

- [Amal Mohammed A Alghamdi](https://github.com/amal-ghamdi)

- [Chao Zhang](https://github.com/chaozg)

- [Jakob Sauer Jørgensen](https://github.com/jakobsj)

## 🌟 Contributors
A big shoutout to our passionate team! Discover the talented individuals behind CUQIpy
[here](https://github.com/CUQI-DTU/CUQIpy/graphs/contributors).

## 🤝 Contributing
We welcome contributions to CUQIpy. Please see our [contributing guidelines](https://cuqi-dtu.github.io/CUQIpy/dev/index.html) for more information.