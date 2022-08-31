import pytest
import numpy as np
import sys
# import sys
sys.path.append('../')
from cuqi.testproblem import Deconvolution1D
from cuqi.distribution import Gaussian, GaussianCov, GMRF, Cauchy_diff, Laplace_diff, LMRF

#All Ns are reduced by a factor of 10 for speed. Best results are obtained by increasing Ns by at least 10 times.
@pytest.mark.parametrize("TP_type, phantom, prior, Ns", 
                         [
                             (Deconvolution1D, "gauss", Gaussian(np.zeros(128), 0.071), 20),
                             (Deconvolution1D, "gauss", GaussianCov(np.zeros(128), 0.005), 20),
                             (Deconvolution1D, "gauss", GMRF(np.zeros(128), 100, 1, "zero"), 20),
                             (Deconvolution1D, "square", LMRF(np.zeros(128), 100, 128, 1, "zero"), 100),
                             (Deconvolution1D, "square", Laplace_diff(np.zeros(128), 0.005), 100),
                             (Deconvolution1D, "square", Cauchy_diff(np.zeros(128), 0.01), 50),
                         ])
def test_TP_BayesianProblem_sample(copy_reference, TP_type, phantom, prior, Ns):
    # SKIP NUTS test if not windows (for now)
    if isinstance(prior, Cauchy_diff) and not sys.platform.startswith('win'):
        pytest.skip("NUTS(Cauchy_diff) regression test is not implemented for this platform")

    np.random.seed(19937)

    # Generate TP using this seed (for data consistency)
    TP = TP_type(dim=prior.dim, phantom=phantom)

    # set the prior of testproblem
    TP.prior = prior

    # Sample posterior
    samples = TP.sample_posterior(Ns=Ns)

    # Extract samples and compute properties
    res = samples.samples    
    med_xpos = np.median(res, axis=1)
    sigma_xpos = res.std(axis=1)
    lo95, up95 = np.percentile(res, [2.5, 97.5], axis=1)

    # Load reference file into temp folder and load
    ref_fname = f"{TP_type.__name__}_{phantom}_{prior.__class__.__name__}_{Ns}"
    #if isinstance(prior, Laplace_diff): #Put the case you want to update for here.
    #    np.savez(ref_fname, median=med_xpos, sigma=sigma_xpos, lo95=lo95, up95=up95) #uncomment to update
    ref_file = copy_reference(f"data/{ref_fname}.npz")
    ref = np.load(ref_file)

    # Check results with reference data
    assert med_xpos == pytest.approx(ref["median"], rel=1e-3, abs=1e-6)
    assert sigma_xpos == pytest.approx(ref["sigma"], rel=1e-3, abs=1e-6)
    assert lo95 == pytest.approx(ref["lo95"], rel=1e-3, abs=1e-6)
    assert up95 == pytest.approx(ref["up95"], rel=1e-3, abs=1e-6)
