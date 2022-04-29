# CUQIpy

[![pipeline status](https://lab.compute.dtu.dk/cuqi/cuqipy/badges/master/pipeline.svg)](https://lab.compute.dtu.dk/cuqi/cuqipy/commits/master)

 Computational Uncertainty Quantification for Inverse Problems in python (CUQIpy) is a python package for modeling and solving inverse problems in a Bayesian inference framework. CUQIpy provides a simple high-level interface to perform UQ analysis of inverse problems, while still allowing full control of the models and methods. The package comes equipped with a number of predefined distributions, samplers, models and test problems and is built to be easily further extended when needed.

You can find the full CUQIpy documentation [here](https://cuqi.gitlab.io/cuqipy/). 

 This software package is part of the [CUQI project](https://www.compute.dtu.dk/english/cuqi) funded by [the Villum Foundation.](https://veluxfoundations.dk/en/forskning/teknisk-og-naturvidenskabelig-forskning)

## Quick Example - UQ in 5 lines
A two dimensional deconvolution example
```python
# Imports
import numpy as np
import matplotlib.pyplot as plt
from cuqi.testproblem import Deconvolution2D
from cuqi.distribution import Laplace_diff, GaussianCov 
from cuqi.problem import BayesianProblem

# Step 1: Model and data
model, data, probInfo = Deconvolution2D.get_components(dim=128, phantom=cuqi.data.grains())

# Step 2: Prior
prior = Laplace_diff(location=np.zeros(model.domain_dim),
                     scale=0.01,
                     bc_type='neumann',
                     physical_dim=2)

# Step 3: Likelihood
likelihood = GaussianCov(mean=model, cov=0.0036**2).to_likelihood(data)

# Step 4: Posterior samples
samples = BayesianProblem(likelihood, prior).sample_posterior(200)

# Step 5: Analysis
probInfo.exactSolution.plot(); plt.title("Exact solution")
data.plot(); plt.title("Data")
samples.plot_mean(); plt.title("Posterior mean")
samples.plot_std(); plt.title("Posterior standard deviation")
```
<img src="docs/_static/img/deconv2D_exact_sol.png" alt="Exact solution" width="360">

<img src="docs/_static/img/deconv2D_data.png" alt="Data" width="360">

<img src="docs/_static/img/deconv2D_post_mean.png" alt="Posterior mean" width="360">

<img src="docs/_static/img/deconv2D_post_std.png" alt="Posterior standard deviation" width="360">

## Getting Started
To run `cuqipy` on your local machine, clone the `cuqipy` repository:

```{r, engine='bash', count_lines}
git clone https://lab.compute.dtu.dk/cuqi/cuqipy.git
```

Then go to the project directory:
```{r, engine='bash', count_lines}
cd cuqipy
```

You can run some demos, for example: 
```{r, engine='bash', count_lines}
cd demos
python demo00_MinimalExample.py 
```

### Required Dependencies
Requirements of cuqipy are listed in `cuqipy/requirements.txt` and can be installed via conda by (while in `cuqipy` directory)
```{r, engine='bash', count_lines}
conda install --file requirements.txt
```
or using pip by
```{r, engine='bash', count_lines}
pip install -r requirements.txt 
```

### Optional Dependencies
- [pytest](https://docs.pytest.org): To run the tests on your local machine
- [ASTRA Toolbox](https://github.com/astra-toolbox/astra-toolbox): For modeling CT problems
- [CIL](https://github.com/TomographicImaging/CIL): For modeling CT problems
- [FEniCS](https://fenicsproject.org): For modeling with FEniCS
- [findiff](https://github.com/maroba/findiff): For some if the PDE based demos

## Running the Tests

To make sure that cuqipy runs as expected on your machine and that all requirements
are met, you can run the tests. While in the project
directory `cuqipy`, run:

```{r, engine='bash', count_lines}
python -m pytest 
```

## Building Documentation

To generate sphinx html documentation in your local machine, 
make sure you have working installation of sphinx and sphinx-rtd-theme. 
Then run the following commands in cuqipy directory:  

```{r, engine='bash', count_lines}
cd docs
sphinx-build -b html . _build
```

Then open docs/_build/index.html using your preferred web browser to browse
cuqipy documentation.

## Contributors

See the list of
[contributors](https://lab.compute.dtu.dk/cuqi/cuqipy/-/graphs/master)
who participated in this project.


