import pytest
import numpy as np

import cuqi


@pytest.mark.parametrize("Ns,prior",
                         [
                             (2000, cuqi.distribution.GMRF(np.zeros(128), 25, 128, 1, "zero"))
                         ])
def test_deblur_bayesian_inversion(copy_reference, Ns, prior):
    np.random.seed(19937)

    deblur = cuqi.testproblem.Deblur()

    norm_f = np.linalg.norm(deblur.f_true)

    # set the prior
    deblur.prior = prior

    res = deblur.sample(Ns=Ns).samples

    med_xpos = np.median(res, axis=1)
    sigma_xpos = res.std(axis=1)
    lo95, up95 = np.percentile(res, [2.5, 97.5], axis=1)

    ref_fname = "deblur_bayesian_inversion"
    # save/update reference data
    #np.savez(ref_fname, median=med_xpos, sigma=sigma_xpos, lo95=lo95, up95=up95)
    # copy refence file to temporary folder
    ref_file = copy_reference(f"data/{ref_fname}.npz")
    # load reference data from file
    ref = np.load(ref_file)

    assert med_xpos == pytest.approx(ref["median"])
    assert sigma_xpos == pytest.approx(ref["sigma"])
    assert lo95 == pytest.approx(ref["lo95"])
    assert up95 == pytest.approx(ref["up95"])

    relerr = round(np.linalg.norm(med_xpos - deblur.f_true)/norm_f*100, 2)


@pytest.mark.parametrize("Ns,prior",
                         [
                             (2000, cuqi.distribution.GMRF(np.zeros(128), 25, 128, 1, "zero"))
                         ])
def test_type1_bayesian_inversion(copy_reference, Ns, prior):
    np.random.seed(19937)

    deblur = cuqi.testproblem.Deblur()

    norm_f = np.linalg.norm(deblur.f_true)

    # RHS: measured data
    b = deblur.data
    # model
    A = deblur.model
    # noise
    e = deblur.noise

    type1 = cuqi.problem.Type1(b, A, e, prior)

    res = type1.sample(Ns=Ns).samples
    med_xpos = np.median(res, axis=1)
    sigma_xpos = res.std(axis=1)
    lo95, up95 = np.percentile(res, [2.5, 97.5], axis=1)

    ref_fname = "type1_bayesian_inversion"
    # uncomment to save/update reference data
    #np.savez(ref_fname, median=med_xpos, sigma=sigma_xpos, lo95=lo95, up95=up95)
    # copy refence file to temporary folder
    ref_file = copy_reference(f"data/{ref_fname}.npz")
    # load reference data from file
    ref = np.load(ref_file)

    assert med_xpos == pytest.approx(ref["median"])
    assert sigma_xpos == pytest.approx(ref["sigma"])
    assert lo95 == pytest.approx(ref["lo95"])
    assert up95 == pytest.approx(ref["up95"])

    relerr = round(np.linalg.norm(med_xpos - deblur.f_true)/norm_f*100, 2)
