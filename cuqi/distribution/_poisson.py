import numpy as np
import scipy.stats as sps
from cuqi.distribution import Distribution

class Poisson(Distribution):
    """ Poisson distribution.

    Defines a Poisson distribution for independent random variables. Each
    variable :math:`x_i` follows the pdf

    .. math::

        P(x_i = k; \\lambda_i) = \\frac{\\lambda_i^k e^{-\\lambda_i}}{k!},
        \\quad k \\in \\{0, 1, 2, \\ldots\\},

    where :math:`\\lambda_i > 0` is the rate (mean) parameter of the
    :math:`i`-th component.

    For a multivariate Poisson with independent components, the joint pdf
    is the product of the individual pdfs.

    Parameters
    ----------
    rate : scalar or ndarray, optional
        The rate parameter :math:`\\lambda` (mean) of the distribution.
        Must be strictly positive. When a scalar is given, all components
        share the same rate.

    Example
    -------
    .. code-block:: python

        import cuqi
        import numpy as np

        # Scalar Poisson (single component)
        p = cuqi.distribution.Poisson(rate=3.5)
        print(p.logpdf(4))   # log P(X=4 | lambda=3.5)

        # Multivariate Poisson with independent components
        p = cuqi.distribution.Poisson(rate=np.array([1.0, 5.0, 10.0]))
        s = p.sample(100)    # draw 100 samples of shape (3,)

    """

    def __init__(self, rate=None, **kwargs):

        # Init from abstract distribution class
        super().__init__(**kwargs)

        self.rate = rate

    def logpdf(self, x):
        """Evaluate the log probability mass function at *x*.

        Parameters
        ----------
        x : int, float, or ndarray
            Point at which to evaluate the log-pdf. Non-negative integers are
            in the support; negative values or non-integers yield ``-inf``.

        Returns
        -------
        float
            Sum of log-pdf values over all components.
        """
        if isinstance(x, (float, int)):
            x = np.array([x])
        return np.sum(sps.poisson.logpmf(x, mu=self.rate))

    def _sample(self, N=1, rng=None):
        """Draw *N* samples from the Poisson distribution.

        Parameters
        ----------
        N : int, optional
            Number of samples to draw (default 1).
        rng : numpy.random.RandomState, optional
            Random number generator for reproducibility.

        Returns
        -------
        ndarray, shape (dim, N)
            Array of non-negative integer samples.
        """
        if rng is not None:
            s = rng.poisson(lam=self.rate, size=(N, self.dim)).T
        else:
            s = np.random.poisson(lam=self.rate, size=(N, self.dim)).T
        return s
