import pytest
from scipy.linalg import cholesky
from scipy.sparse import diags
from cuqi.utilities import sparse_cholesky, plot_1D_density, plot_2D_density, count_nonzero, count_constant_components_1D, count_constant_components_2D
from cuqi.model import LinearModel
from cuqi.distribution import Gaussian, Uniform, JointDistribution
import numpy as np
import matplotlib.pyplot as plt


@pytest.mark.parametrize("P", [
    diags([-1, 2, -1], [-1, 0, 1], shape=(128, 128)),
    diags([1, -4, 6, -4, 1], [-2, -1, 0, 1, 2], shape=(128, 128))
])
def test_sparse_cholesky(P):
    """ Test the sparse Cholesky decomposition. P is a sparse matrix (often precision). """
    # Scipy version (on dense only)
    L1 = cholesky(P.toarray())

    # Scipy-based version from CUQIpy (on sparse)
    L2 = sparse_cholesky(P) 

    assert np.allclose(L1, L2.toarray()) # Convert to dense to compare


def posterior_2D_components():
    """A function to create and return 2D Bayesian posterior components"""
    # Create a CUQIpy model
    A = LinearModel(np.array([[1.0, 1.0]]))
    # Create a CUQIpy prior
    x = Gaussian(mean=np.array([0.0, 0.0]), cov=np.array([1.0, 1.0]))
    # Create a CUQIpy likelihood
    y = Gaussian(A@x, cov=0.1)
    return x, y


def likelihood_2D():
    """A function to create and return a 2D likelihood"""
    x, y = posterior_2D_components()
    likelihood = y.to_likelihood(2)
    return likelihood


def posterior_2D():
    """A function to create and return a 2D posterior"""
    x, y = posterior_2D_components()
    joint = JointDistribution(x, y)
    posterior = joint(y=2)
    return posterior


@pytest.mark.parametrize("density", [
    Gaussian(mean=0.0, cov=1.0),
    Uniform(low=0.0, high=1.0),
    Gaussian(mean=np.array([0.0, 0.0]), cov=np.array([1.0, 1.0])),
    likelihood_2D(),
    posterior_2D()
])
@pytest.mark.parametrize("log_scale", [True, False])
def test_plot_density(density, log_scale):
    """ Test the 1D and 2D density plot. """
    if density.dim == 1:
        plot_1D_density(density, -1, 1, color='green', log_scale=log_scale)
    elif density.dim == 2:
        plot_2D_density(
            density, -3, 3, -3, 3, 60, 60, cmap='gray', log_scale=log_scale)
    else:
        raise ValueError("Density must be 1D or 2D.")
    
def count_nonzero_threshold():
    threshold = 1e-8
    eps_min = 1e-1
    eps_plus = 1e1
    test_array = np.array([0, -eps_min*threshold, -threshold, -eps_plus*threshold, eps_min*threshold, threshold, eps_plus*threshold])
    print(test_array)

    nonzeros = count_nonzero(test_array, threshold)
    assert(nonzeros == 4)

def count_constant_components_1d_threshold():
    threshold = 1e-4
    test_array = np.array([0, 0, 1.0 - 0.5*threshold, 1.0, 1.0+0.5*threshold, 1.5, 2, 2+0.5*threshold, 2+threshold, 0, 0])

    components = count_constant_components_1D(test_array, threshold)
    assert(components == 5)

    components = count_constant_components_1D(test_array, threshold, lower = 0, upper = 2)
    assert(components == 2)

def count_constant_components_2d_threshold():
    threshold = 1e-4
    test_array = np.array([[0, 0                , 1 - 0.25*threshold],
                           [0, 1                , 1 + 0.25*threshold],
                           [2, 2 - 0.5*threshold, 2]])

    components = count_constant_components_2D(test_array, threshold)
    assert(components == 3)

    components = count_constant_components_2D(test_array, threshold, lower = 0, upper = 2)
    assert(components == 1)