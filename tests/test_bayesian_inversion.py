from typing import Dict

import pytest
import numpy as np
import sys

from cuqi.testproblem import Deconvolution1D
from cuqi.distribution import Gaussian, GMRF, CMRF, LMRF, Gamma
from cuqi.implicitprior import RegularizedGaussian, RegularizedGMRF
from cuqi.problem import BayesianProblem
from cuqi.density import Density

#All Ns are reduced by a factor of 10 for speed. Best results are obtained by increasing Ns by at least 10 times.
@pytest.mark.parametrize("TP_type, phantom, prior, Ns, use_legacy", 
                         [
                             (Deconvolution1D, "gauss", Gaussian(np.zeros(128), 0.071**2), 20, True),
                             (Deconvolution1D, "gauss", GMRF(np.zeros(128), 100, "zero"), 20, True),
                             (Deconvolution1D, "square", LMRF(0, 0.005, geometry=128), 100, True),
                             (Deconvolution1D, "square", CMRF(np.zeros(128), 0.01), 50, True),
                             (Deconvolution1D, "square", RegularizedGaussian(np.zeros(128), 0.1, constraint="nonnegativity"), 100, False),
                             (Deconvolution1D, "square", RegularizedGMRF(np.zeros(128), 50, constraint="nonnegativity"), 100, False),
                         ])
@pytest.mark.parametrize("experimental", [False, True])
def test_TP_BayesianProblem_sample(copy_reference, TP_type, phantom, prior, Ns, use_legacy, experimental):
    # SKIP NUTS test if not windows (for now)
    if isinstance(prior, CMRF) and not sys.platform.startswith('win'):
        pytest.skip("NUTS(CMRF) regression test is not implemented for this platform")

    np.random.seed(19937)

    # Generate TP using this seed (for data consistency)
    # Legacy convolution is used for consistency with the reference data.
    TP = TP_type(dim=prior.dim, phantom=phantom, use_legacy=use_legacy, noise_std=0.05) 

    # set the prior of testproblem
    TP.prior = prior

    # Sample posterior
    samples = TP.sample_posterior(Ns=Ns, experimental=experimental)

    # Extract samples and compute properties
    res = samples.samples    
    med_xpos = np.median(res, axis=1)
    sigma_xpos = res.std(axis=1)
    lo95, up95 = np.percentile(res, [2.5, 97.5], axis=1)

    # Load reference file into temp folder and load
    if experimental:
        ref_fname = f"{TP_type.__name__}_{phantom}_{prior.__class__.__name__}_{Ns}_experimental"
    else:
        ref_fname = f"{TP_type.__name__}_{phantom}_{prior.__class__.__name__}_{Ns}"

    #if isinstance(prior, RegularizedGMRF): #Put the case you want to update for here.
    #    np.savez(ref_fname, median=med_xpos, sigma=sigma_xpos, lo95=lo95, up95=up95) #uncomment to update
    
    ref_file = copy_reference(f"data/{ref_fname}.npz")
    ref = np.load(ref_file)

    # Check results with reference data
    assert med_xpos == pytest.approx(ref["median"], rel=1e-3, abs=1e-6)
    assert sigma_xpos == pytest.approx(ref["sigma"], rel=1e-3, abs=1e-6)
    assert lo95 == pytest.approx(ref["lo95"], rel=1e-3, abs=1e-6)
    assert up95 == pytest.approx(ref["up95"], rel=1e-3, abs=1e-6)

@pytest.mark.parametrize("experimental", [False, True])
@pytest.mark.parametrize("TP_type, phantom, priors, Ns",
    [
        # Case: Gaussian prior (no hyperparameters)
        (
            Deconvolution1D,
            "gauss",
            [
                Gaussian(np.zeros(128), 0.005, name="x")
            ],
            50
        ),
        # Case: Gaussian with Gamma hyperprior on noise precision
        (
            Deconvolution1D,
            "gauss",
            [
                Gaussian(np.zeros(128), 0.005, name="x"),
                Gamma(1, 1e-4, name="l")
            ],
            50
        ),
        # Case: Gaussian with Gamma hyperpriors on both noise and prior precision
        (
            Deconvolution1D,
            "gauss",
            [
                Gaussian(np.zeros(128), lambda d: 1/d, name="x"),
                Gamma(1, 1e-4, name="l"),
                Gamma(1, 1e-4, name="d")
            ],
            50
        ),
        # Case 2: LMRF with Gamma hyperpriors on both noise and prior precision
        (
            Deconvolution1D,
            "square",
            [
                LMRF(0, lambda d: 1/d, geometry=128, name="x"),
                Gamma(1, 1e-4, name="l"),
                Gamma(1, 1e-4, name="d")
            ],
            50,
        ),
        # Case: RegularizedGaussian with Gamma hyperpriors on both noise and prior precision
        (
            Deconvolution1D,
            "square",
            [
                RegularizedGaussian(np.zeros(128), lambda d: 1/d, constraint="nonnegativity", name="x"),
                Gamma(1, 1e-4, name="l"),
                Gamma(1, 1e-4, name="d")
            ],
            50,
        ),
        # Case: RegularizedGMRF with Gamma hyperpriors on both noise and prior precision
        (
            Deconvolution1D,
            "square",
            [
                RegularizedGMRF(np.zeros(128), lambda d: d, constraint="nonnegativity", name="x"),
                Gamma(1, 1e-4, name="l"),
                Gamma(1, 1e-4, name="d")
            ],
            50,
        ),
    ]
)
def test_Bayesian_inversion_hierarchical(TP_type: BayesianProblem, phantom: str, priors: Dict[str, Density], Ns: int, experimental: bool):
    """ This tests Bayesian inversion for Bayesian Problem using a hierarchical model.
    
    It is an end-to-end test that checks that the posterior samples are consistent with the expected shape.

    Currently, no reference data is available for this test, so no regression test is performed.
    """
    # Load model + data from testproblem library
    A, y_data, probInfo = TP_type(dim=priors[0].dim, phantom=phantom).get_components()

    # data distribution
    if len(priors) == 1: # No hyperparameters
        data_dist = Gaussian(A@priors[0], 400, name="y")
    else:
        data_dist = Gaussian(A@priors[0], lambda l: 1/l, name="y")

    # Bayesian problem
    BP = BayesianProblem(data_dist, *priors).set_data(y=y_data)

    # Sample posterior using UQ method
    if len(priors) == 1: # No hyperparameters
        samples = BP.UQ(Ns=Ns, exact=probInfo.exactSolution, experimental=experimental)
    else:
        samples = BP.UQ(Ns=Ns, exact={priors[0].name: probInfo.exactSolution}, experimental=experimental)

    # No regression test yet, just check that the samples are the right shape
    if isinstance(samples, dict): # Gibbs case
        for prior in priors:
            assert samples[prior.name].shape == (prior.dim, Ns)
    else:
        assert samples.shape == (priors[0].dim, Ns)
