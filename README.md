# CUQIpy

[![pipeline status](https://lab.compute.dtu.dk/cuqi/cuqipy/badges/master/pipeline.svg)](https://lab.compute.dtu.dk/cuqi/cuqipy/commits/master)

 Computational Uncertainty Quantification for Inverse Problems python package (CUQIpy) is a python package for modeling and solving inverse problems in a Bayesian inference framework. CUQIpy is an easy-to-use tool that allows non-experts in UQ and Bayesian inversion to perform UQ analysis of inverse problems. At the same time, it allows expert users full control of the models and methods. The package is equipped with a number of predefined 1D and 2D test problem.

 This software package is funded by [the Villum Foundation](https://veluxfoundations.dk/en/forskning/teknisk-og-naturvidenskabelig-forskning) as part of the [CUQI project.](https://www.compute.dtu.dk/english/cuqi)

## Quick Example
A two dimensional deconvolution example
```python
#%% Imports
import cuqi
import matplotlib.pyplot as plt

#%% Load testproblem
TP = cuqi.testproblem.Deconvolution2D(phantom=cuqi.data.grains(256))

#%% Plot the exact solution we want to infer
TP.exactSolution.plot()
plt.title("Exact solution")

#%% Plot the data we use in the Bayesain inversion
TP.data.plot()
plt.title("Data")

#%% Add prior
TP.prior = cuqi.distribution.Laplace_diff(location=np.zeros(TP.model.domain_dim),
                                          scale=0.01,
                                          bc_type='neumann',
                                          physical_dim=2)

#%% Now sample the posterior
post_samples = TP.sample_posterior(200)

#%% Plot the samples mean
post_samples.plot_mean()

#%% plot the samples standard deviation
post_samples.plot_std()
```
<img src="/uploads/1c7b4adf09f6a71405f37cdd44016bec/deconv2D_exact_sol.png"  width="360">

<img src="/uploads/d7a041ae89afdfe0f4a31723305813fc/deconv2D_data.png"  width="360">

<img src="/uploads/6e919bca40ab8136b064d76128d9f0ac/deconv2D_mean.png"  width="360">

<img src="/uploads/712861b89cda91fd06fa7e364116ebe5/deconv2D_std.png"  width="360">

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

### Requirements
Requirements of cuqipy are listed in `requirements.txt` and can be installed by
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

To make sure that cuqipy runs on your machine and that all requirements
are met, you can run the tests. While in the project
directory `cuqipy`, run:

```{r, engine='bash', count_lines}
python -m pytest -s -v tests/test_distribution.py 
```

## Documentation

You can find CUQIpy documentation [here](https://cuqi.gitlab.io/cuqipy/). 

Alternatively, to generate sphinx html documentation in your local machine, 
make sure you have working installation of sphinx and sphinx-rtd-theme. 
Then run the following commands in cuqipy directory:  

```{r, engine='bash', count_lines}
cd docs
sphinx-build -b html . _build
```

Then open docs/_build/index.html using your preferred web browser to browse
cuqipy documentation.

## Authors

See the list of
[contributors](https://lab.compute.dtu.dk/cuqi/cuqipy/-/graphs/master)
who participated in this project.


