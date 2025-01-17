from cuqi.utilities import get_non_default_args
from cuqi.distribution import Distribution, Gaussian
from cuqi.solver import ProjectNonnegative, ProjectBox, ProximalL1
from cuqi.geometry import Continuous1D, Continuous2D, Image2D
from cuqi.operator import FirstOrderFiniteDifference

import numpy as np
from copy import copy


class RegularizedGaussian(Distribution):
    """ Implicit Regularized Gaussian.

    Defines a so-called implicit prior based on a Gaussian distribution with implicit regularization.
    The regularization can be defined in the form of a proximal operator or a projector. 
    Alternatively, preset constraints and regularization can be used.

    Precisely one of proximal, projector, constraint or regularization needs to be provided. Otherwise, an error is raised.

    Can be used as a prior in a posterior which can be sampled with the RegularizedLinearRTO sampler.


    For more details on implicit regularized Gaussian see the following paper:

    [1] Everink, Jasper M., Yiqiu Dong, and Martin S. Andersen. "Sparse Bayesian inference with regularized
    Gaussian distributions." Inverse Problems 39.11 (2023): 115004.

    Parameters
    ----------
    mean
        See :class:`~cuqi.distribution.Gaussian` for details.

    cov
        See :class:`~cuqi.distribution.Gaussian` for details.

    prec
        See :class:`~cuqi.distribution.Gaussian` for details.

    sqrtcov
        See :class:`~cuqi.distribution.Gaussian` for details.

    sqrtprec
        See :class:`~cuqi.distribution.Gaussian` for details.

    proximal : callable f(x, scale), list of tuples (callable proximal operator of f_i, linear operator L_i) or None
        If callable:
            Euclidean proximal operator f of the regularization function g, that is, a solver for the optimization problem
            min_z 0.5||x-z||_2^2+scale*g(x).
        If list of tuples (callable proximal operator of f_i, linear operator L_i):
            Each callable proximal operator of f_i accepts two arguments (x, p) and should return the minimizer of p/2||x-z||^2 + f(x) over z for some f.
            The corresponding regularization takes the form
                sum_i f_i(L_i x),
            where the sum ranges from 1 to an arbitrary n.

    projector : callable f(x) or None
        Euclidean projection onto the constraint C, that is, a solver for the optimization problem
        min_(z in C) 0.5||x-z||_2^2.

    constraint : string or None
        Preset constraints that generate the corresponding proximal parameter. Can be set to "nonnegativity" and "box". Required for use in Gibbs.
        For "box", the following additional parameters can be passed:
            lower_bound : array_like or None
                Lower bound of box, defaults to zero
            upper_bound : array_like
                Upper bound of box, defaults to one

    regularization : string or None
        Preset regularization that generate the corresponding proximal parameter. Can be set to "l1" or 'tv'. Required for use in Gibbs in future update.
        For "l1" or "tv", the following additional parameters can be passed:
            strength : scalar
                Regularization parameter, i.e., strength*||Lx||_1, defaults to one

    """
        
    def __init__(self, mean=None, cov=None, prec=None, sqrtcov=None, sqrtprec=None, proximal = None, projector = None, constraint = None, regularization = None, **kwargs):
        
        # Store regularization parameters and remove them from kwargs passed to Gaussian
        optional_regularization_parameters = {
            "lower_bound" : kwargs.pop("lower_bound", None), # Takes default of ProjectBox if None
            "upper_bound" : kwargs.pop("upper_bound", None), # Takes default of ProjectBox if None
            "strength" : kwargs.pop("strength", 1)
        }
        
        # We init the underlying Gaussian first for geometry and dimensionality handling
        self._gaussian = Gaussian(mean=mean, cov=cov, prec=prec, sqrtcov=sqrtcov, sqrtprec=sqrtprec, **kwargs)
        kwargs.pop("geometry", None)

        # Init from abstract distribution class
        super().__init__(**kwargs)

        self._parse_regularization_input_arguments(proximal, projector, constraint, regularization, optional_regularization_parameters)

    def _parse_regularization_input_arguments(self, proximal, projector, constraint, regularization, optional_regularization_parameters):
        """ Parse regularization input arguments with guarding statements and store internal states """

        # Check that only one of proximal, projector, constraint or regularization is provided        
        if (proximal is not None) + (projector is not None) + (constraint is not None) + (regularization is not None) != 1:
            raise ValueError("Precisely one of proximal, projector, constraint or regularization needs to be provided.")

        if projector is not None:
            if not callable(projector):
                raise ValueError("Projector needs to be callable.")
            if len(get_non_default_args(projector)) != 1:
                raise ValueError("Projector should take 1 argument.")
            
        # Preset information, for use in Gibbs
        self._preset = None
        
        if proximal is not None:
            # No need to generate the proximal and associated information
            self.proximal = proximal
        elif projector is not None:
            self._proximal = lambda z, gamma: projector(z)
        elif (isinstance(constraint, str) and constraint.lower() == "nonnegativity"):
            self._proximal = lambda z, gamma: ProjectNonnegative(z)
            self._preset = "nonnegativity"
            self._box_bounds = (np.ones(self.dim)*0, np.ones(self.dim)*np.inf)
        elif (isinstance(constraint, str) and constraint.lower() == "box"):
            self._box_lower = optional_regularization_parameters["lower_bound"]
            self._box_upper = optional_regularization_parameters["upper_bound"]
            self._box_bounds = (np.ones(self.dim)*self._box_lower, np.ones(self.dim)*self._box_upper)
            self._proximal = lambda z, _: ProjectBox(z, self._box_lower, self._box_upper)
            self._preset = "box" # Not supported in Gibbs
        elif (isinstance(regularization, str) and regularization.lower() in ["l1"]):
            self._strength = optional_regularization_parameters["strength"]
            self._proximal = lambda z, gamma: ProximalL1(z, gamma*self._strength)
            self._preset = "l1"
        elif (isinstance(regularization, str) and regularization.lower() in ["tv"]):
            self._strength = optional_regularization_parameters["strength"]
            if isinstance(self.geometry, (Continuous1D, Continuous2D, Image2D)):
                self._transformation = FirstOrderFiniteDifference(self.geometry.fun_shape, bc_type='neumann')
            else:
                raise ValueError("Geometry not supported for total variation")
            
            self._regularization_prox = lambda z, gamma: ProximalL1(z, gamma*self._strength)
            self._regularization_oper = self._transformation

            self._proximal = [(self._regularization_prox, self._regularization_oper)]
            self._preset = "tv"
        else:
            raise ValueError("Regularization not supported")

    
    @property
    def transformation(self):
        return self._transformation
    
    @property
    def strength(self):
        return self._strength
        
    @strength.setter
    def strength(self, value):
        if self._preset not in self.regularization_options():
            raise TypeError("Strength is only used when the regularization is set to l1 or TV.")

        self._strength = value
        if self._preset == "tv":
            self._regularization_prox = lambda z, gamma: ProximalL1(z, gamma*self._strength)
            self._proximal = [(self._regularization_prox, self._regularization_oper)]
        elif self._preset == "l1":
            self._proximal = lambda z, gamma: ProximalL1(z, gamma*self._strength)

    # This is a getter only attribute for the underlying Gaussian
    # It also ensures that the name of the underlying Gaussian
    # matches the name of the implicit regularized Gaussian
    @property
    def gaussian(self):
        if self._name is not None:
            self._gaussian._name = self._name
        return self._gaussian
    
    @property
    def proximal(self):
        return self._proximal
    
    @proximal.setter
    def proximal(self, value):
        if callable(value):
            if len(get_non_default_args(value)) != 2:
                raise ValueError("Proximal should take 2 arguments.")
        elif isinstance(value, list):
            for (prox, op) in value:
                if len(get_non_default_args(prox)) != 2:
                    raise ValueError("Proximal should take 2 arguments.")
                if op.shape[1] != self.geometry.par_dim:
                    raise ValueError("Incorrect shape of linear operator in proximal list.")
        else:
            raise ValueError("Proximal needs to be callable or a list. See documentation.")
        
        self._proximal = value

        # For all the presets, self._proximal is set directly, 
        self._preset = None
            
    @property
    def preset(self):
        return self._preset

    def logpdf(self, x):
        return np.nan
        #raise ValueError(
        #    f"The logpdf of a implicit regularized Gaussian is not be defined.")
        
    def _sample(self, N, rng=None):
        raise ValueError(
            "Cannot be sampled from.")
  
    @staticmethod
    def constraint_options():
        return ["nonnegativity", "box"]

    @staticmethod
    def regularization_options():
        return ["l1", "tv"]


    # --- Defer behavior of the underlying Gaussian --- #
    @property
    def geometry(self):
        return self.gaussian.geometry
    
    @geometry.setter
    def geometry(self, value):
        self.gaussian.geometry = value
    
    @property
    def mean(self):
        return self.gaussian.mean
    
    @mean.setter
    def mean(self, value):
        self.gaussian.mean = value
    
    @property
    def cov(self):
        return self.gaussian.cov
    
    @cov.setter
    def cov(self, value):
        self.gaussian.cov = value
    
    @property
    def prec(self):
        return self.gaussian.prec
    
    @prec.setter
    def prec(self, value):
        self.gaussian.prec = value
    
    @property
    def sqrtprec(self):
        return self.gaussian.sqrtprec
    
    @sqrtprec.setter
    def sqrtprec(self, value):
        self.gaussian.sqrtprec = value
    
    @property
    def sqrtcov(self):
        return self.gaussian.sqrtcov
    
    @sqrtcov.setter
    def sqrtcov(self, value):
        self.gaussian.sqrtcov = value     
    
    def get_mutable_variables(self):
        mutable_vars = self.gaussian.get_mutable_variables().copy()
        if self.preset in self.regularization_options():
            mutable_vars += ["strength"]
        return mutable_vars
    
    def _make_copy(self):
        """ Returns a shallow copy of the density keeping a pointer to the original. """
        # Using deepcopy would also copy the underlying geometry, which causes a crash because geometries won't match anymore.
        new_density = copy(self)
        new_density._gaussian = copy(new_density._gaussian)
        new_density._original_density = self
        return new_density


class ConstrainedGaussian(RegularizedGaussian):
    """ Implicit Constrained Gaussian.

    Defines a so-called implicit prior based on a Gaussian distribution with implicit constraints.
    The constraint can be defined as a preset or in the form of a projector. 

    Precisely one of projector or constraint needs to be provided. Otherwise, an error is raised.

    Can be used as a prior in a posterior which can be sampled with the RegularizedLinearRTO sampler.

    Alias for :class:`~cuqi.implicitprior.RegularizedGaussian` with only constraints available.

    For more details on implicit regularized Gaussian see the following paper:

    [1] Everink, Jasper M., Yiqiu Dong, and Martin S. Andersen. "Sparse Bayesian inference with regularized
    Gaussian distributions." Inverse Problems 39.11 (2023): 115004.

    Parameters
    ----------
    mean
        See :class:`~cuqi.distribution.Gaussian` for details.

    cov
        See :class:`~cuqi.distribution.Gaussian` for details.

    prec
        See :class:`~cuqi.distribution.Gaussian` for details.

    sqrtcov
        See :class:`~cuqi.distribution.Gaussian` for details.

    sqrtprec
        See :class:`~cuqi.distribution.Gaussian` for details.

    projector : callable f(x) or None
        Euclidean projection onto the constraint C, that is, a solver for the optimization problem
        min_(z in C) 0.5||x-z||_2^2.

    constraint : string or None
        Preset constraints that generate the corresponding proximal parameter. Can be set to "nonnegativity" and "box". Required for use in Gibbs.
        For "box", the following additional parameters can be passed:
            lower_bound : array_like or None
                Lower bound of box, defaults to zero
            upper_bound : array_like
                Upper bound of box, defaults to one

    """
    def __init__(self, mean=None, cov=None, prec=None, sqrtcov=None,sqrtprec=None, projector=None, constraint=None, **kwargs):
        super().__init__(mean=mean, cov=cov, prec=prec, sqrtcov=sqrtcov, sqrtprec=sqrtprec, projector=projector, constraint=constraint, **kwargs)

        
class NonnegativeGaussian(RegularizedGaussian):
    """ Implicit Nonnegative Gaussian.

    Defines a so-called implicit prior based on a Gaussian distribution with implicit nonnegativity constraints.

    Can be used as a prior in a posterior which can be sampled with the RegularizedLinearRTO sampler.

    Alias for :class:`~cuqi.implicitprior.RegularizedGaussian` with only nonnegativity constraints.

    For more details on implicit regularized Gaussian see the following paper:

    [1] Everink, Jasper M., Yiqiu Dong, and Martin S. Andersen. "Sparse Bayesian inference with regularized
    Gaussian distributions." Inverse Problems 39.11 (2023): 115004.

    Parameters
    ----------
    mean
        See :class:`~cuqi.distribution.Gaussian` for details.

    cov
        See :class:`~cuqi.distribution.Gaussian` for details.

    prec
        See :class:`~cuqi.distribution.Gaussian` for details.

    sqrtcov
        See :class:`~cuqi.distribution.Gaussian` for details.

    sqrtprec
        See :class:`~cuqi.distribution.Gaussian` for details.

    """
    def __init__(self, mean=None, cov=None, prec=None, sqrtcov=None,sqrtprec=None, **kwargs):
        super().__init__(mean=mean, cov=cov, prec=prec, sqrtcov=sqrtcov, sqrtprec=sqrtprec, constraint="nonnegativity", **kwargs)