import pytest
from scipy.linalg import cholesky
from scipy.sparse import diags
from cuqi.utilities import sparse_cholesky
import numpy as np


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
