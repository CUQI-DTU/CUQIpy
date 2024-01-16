from cuqi.implicitprior import RegularizedGaussian
from cuqi.distribution import Distribution, GMRF

class RegularizedGMRF(RegularizedGaussian):
    """ Implicit Regularized GMRF (Gaussian Markov Random Field). 

    Defines a GMRF distribution with implicit regularization defining a so-called implicit prior.
    The regularization can be defined in the form of a proximal operator or a projector.
    Alternatively, preset constraints and regularization can be used.

    Only one of proximal, projector, constraint or regularization can be provided. If none of them are provided,
    a nonnegativity constraint is used by default.

    Can be used as a prior in a posterior which can be sampled with the RegularizedLinearRTO sampler.


    For more details on implicit regularized Gaussian see the following paper:

    [1] Everink, Jasper M., Yiqiu Dong, and Martin S. Andersen. "Sparse Bayesian inference with regularized
    Gaussian distributions." Inverse Problems 39.11 (2023): 115004.

    Parameters
    ----------
    mean
        See :class:`~cuqi.distribution.GMRF` for details.
        
    prec
        See :class:`~cuqi.distribution.GMRF` for details.

    physical_dim
        See :class:`~cuqi.distribution.GMRF` for details.

    bc_type
        See :class:`~cuqi.distribution.GMRF` for details.

    order
        See :class:`~cuqi.distribution.GMRF` for details.

    proximal : callable f(x, scale) or None
        Euclidean proximal operator f of the regularization function g, that is, a solver for the optimization problem
        min_z 0.5||x-z||_2^2+scale*g(x).

    projector : callable f(x) or None
        Euclidean projection onto the constraint C, that is, a solver for the optimization problem
        min_(z in C) 0.5||x-z||_2^2.

    constraint : string or None
        Preset constraints, including "nonnegativity" and "box". Required for use in Gibbs.

    regularization : string or None
        Preset regularization, including "l1". Required for use in Gibbs in future update.

    """
    # TODO: Once GMRF is updated, add default None to mean and prec here.
    def __init__(self, mean, prec, physical_dim=1, bc_type='zero', order=1, proximal = None, projector = None, constraint = None, regularization = None, **kwargs):
            
            args = {"lower_bound" : kwargs.pop("lower_bound", None),
                    "upper_bound" : kwargs.pop("upper_bound", None),
                    "strength" : kwargs.pop("strength", None)}
            
            # Underlying explicit Gaussian
            self._gaussian = GMRF(mean, prec, physical_dim=physical_dim, bc_type=bc_type, order=order, **kwargs)
            
            # Init from abstract distribution class
            super(Distribution, self).__init__(**kwargs)

            self._parse_regularization_input_arguments(proximal, projector, constraint, regularization, args)
