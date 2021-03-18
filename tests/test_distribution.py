import cuqi
import numpy as np

from pytest import approx

def test_Normal_mean_standard():
    assert cuqi.distribution.Normal(0,1).mean == approx(0.0)

def test_Normal_pdf_mean():
    pX = cuqi.distribution.Normal(0.1,1)
    assert pX.pdf(0.1) == approx(1.0/np.sqrt(2.0*np.pi))
