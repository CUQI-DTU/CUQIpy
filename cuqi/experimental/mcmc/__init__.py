""" Re-implementation of sampler module in a more object oriented way. """

from ._sampler import SamplerNew, ProposalBasedSamplerNew
from ._langevin_algorithm import ULANew, MALANew
from ._mh import MHNew
from ._pcn import PCNNew
from ._rto import LinearRTONew, RegularizedLinearRTONew
from ._cwmh import CWMHNew
from ._laplace_approximation import UGLANew
from ._hmc import NUTSNew
from ._gibbs import HybridGibbsNew
from ._conjugate import ConjugateNew
from ._conjugate_approx import ConjugateApproxNew
from ._direct import DirectNew
