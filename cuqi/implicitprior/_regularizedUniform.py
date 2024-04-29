from cuqi.implicitprior import RegularizedGaussian
from cuqi.distribution import Distribution, Gaussian

import numpy as np


class RegularizedUniform(RegularizedGaussian):
        
    def __init__(self, proximal = None, projector = None, constraint = None, regularization = None, geometry = None, **kwargs):
        
        args = {"lower_bound" : kwargs.pop("lower_bound", None),
                 "upper_bound" : kwargs.pop("upper_bound", None),
                 "strength" : kwargs.pop("strength", None)}
            
        # Underlying explicit Gaussian
        self._gaussian = Gaussian(mean = np.zeros(geometry.par_dim), sqrtprec = np.zeros((geometry.par_dim,geometry.par_dim)), **kwargs) 
        kwargs.pop("geometry", None)

        # Init from abstract distribution class
        super(Distribution, self).__init__(**kwargs)

        self._parse_regularization_input_arguments(proximal, projector, constraint, regularization, args)
