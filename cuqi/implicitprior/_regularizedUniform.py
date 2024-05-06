from cuqi.implicitprior import RegularizedGaussian
from cuqi.distribution import Distribution, Gaussian

import numpy as np


class RegularizedUniform(RegularizedGaussian):
        """ Implicit Regularized GMRF (Gaussian Markov Random Field). 

        Defines a so-called implicit prior based on a Gaussian distribution with implicit regularization and zero precision.
        The regularization can be defined in the form of a proximal operator or a projector.
        Alternatively, preset constraints and regularization can be used.

        Only one of proximal, projector, constraint or regularization can be provided. Otherwise, an error is raised.

        Can be used as a prior in a posterior which can be sampled with the RegularizedLinearRTO sampler.


        For more details on implicit regularized Gaussian see the following paper:

        [1] Everink, Jasper M., Yiqiu Dong, and Martin S. Andersen. "Sparse Bayesian inference with regularized
        Gaussian distributions." Inverse Problems 39.11 (2023): 115004.

        Parameters
        ----------
        proximal : callable f(x, scale) or None
                Euclidean proximal operator f of the regularization function g, that is, a solver for the optimization problem
                min_z 0.5||x-z||_2^2+scale*g(x).

        regularization : string or None
                Preset regularization. Can be set to "l1". Required for use in Gibbs in future update.
                For "l1", the following additional parameters can be passed:
                strength : scalar
                        Regularization parameter, i.e., strength*||x||_1 , defaults to one

                For "TV", the following additional parameters can be passed:
                strength : scalar
                        Regularization parameter, i.e., strength*||x||_TV , defaults to one

        """
        def __init__(self, proximal = None , regularization = None, geometry = None, **kwargs):
                
                args = {"lower_bound" : kwargs.pop("lower_bound", None),
                        "upper_bound" : kwargs.pop("upper_bound", None),
                        "strength" : kwargs.pop("strength", None)}
                
                # Underlying explicit Gaussian
                
                # This line throws a warning to due trying to applying get_sqrtprec_from_sqrtprec to an all zero matrix  
                self._gaussian = Gaussian(mean = np.zeros(geometry.par_dim), sqrtprec = np.zeros((geometry.par_dim,geometry.par_dim)), geometry = geometry, **kwargs) 

                # Init from abstract distribution class
                super(Distribution, self).__init__(**kwargs)

                self._parse_regularization_input_arguments(proximal, None, None, regularization, args)


        def get_conditioning_variables(self):
                return super(RegularizedGaussian, self).get_conditioning_variables()
        
        def get_mutable_variables(self):
                # The following line will still return all Gaussian parameters
                #return super(RegularizedGaussian, self).get_mutable_variables()
                # Hence we manually return them.
                if self.preset in ["l1", "TV"]:
                        return ["strength"]
                else:
                        return []

        # Revert back to the original conditioning, as this underlying Gaussian should not be modified.
        def _condition(self, *args, **kwargs):
                return super(RegularizedGaussian, self)._condition(*args, **kwargs)