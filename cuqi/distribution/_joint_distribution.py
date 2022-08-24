from __future__ import annotations
from typing import List
from copy import copy
from cuqi.density import Density, EvaluatedDensity
from cuqi.distribution import Distribution, Posterior
from cuqi.likelihood import Likelihood
from cuqi.geometry import Geometry, _DefaultGeometry
import numpy as np # for splitting array. Can avoid.

class JointDistribution:
    """ 
    Joint distribution of multiple variables.

    Parameters
    ----------
    densities: List of Density objects
        The densities can be Distribution, Likelihood, EvaluatedDensity etc.

    Notes
    -----
    The joint distribution allows conditioning on any variable of the distribution.
    This is useful for example when Gibbs sampling and the conditionals are needed.
    Conditioning essentially fixes the variable and stores its contribution to the
    log density function.

    Example
    -------

    Consider defining the joint distribution:

    .. math::

        p(y,x,z) = p(y \mid x)p(x \mid z)p(z)

    and conditioning on :math:`y=y_{obs}` leading to the posterior:

    .. math::

        p(x,z \mid y_{obs}) = p(y_{obs} \mid x)p(x \mid z)p(z)

    .. code-block:: python

        import cuqi
        import numpy as np

        y_obs = np.random.randn(10)
        A = np.random.randn(10,3)

        # Define distributions for Bayesian model
        y = cuqi.distribution.Normal(lambda x: A@x, np.ones(10))
        x = cuqi.distribution.Normal(np.zeros(3), lambda z:z)
        z = cuqi.distribution.Gamma(1, 1)

        # Joint distribution p(y,x,z)
        joint = cuqi.distribution.JointDistribution([y, x, z])

        # Posterior p(x,z | y_obs)
        posterior = joint(y=y_obs)
        
    """
    def __init__(self, densities: List[Density]):

        # Ensure all densities have unique names
        names = [density.name for density in densities]
        if len(names) != len(set(names)):
            raise ValueError("All densities must have unique names.")

        self.densities = densities
        self._allow_reduce = False # Hack to allow conditioning to reduce a joint distribution to a single density

        # Make sure every unspecified parameter has a distribution (prior)
        cond_vars = self.get_conditioning_variables()
        if len(cond_vars) > 0:
            raise ValueError(f"Every density parameter must have a distribution (prior). Missing prior for {cond_vars}.")

    @property
    def distributions(self) -> List[Distribution]:
        """ Returns a list of the distributions (priors) in the joint distribution. """
        return [dist for dist in self.densities if isinstance(dist, Distribution)]

    @property
    def likelihoods(self) -> List[Likelihood]:
        """ Returns a list of the likelihoods in the joint distribution. """
        return [likelihood for likelihood in self.densities if isinstance(likelihood, Likelihood)]

    @property
    def evaluated_densities(self) -> List[EvaluatedDensity]:
        """ Returns a list of the evaluated densities in the joint distribution. """
        return [eval_dens for eval_dens in self.densities if isinstance(eval_dens, EvaluatedDensity)]

    @property
    def dim(self) -> List[int]:
        """ Returns the dimensions of the joint distribution. """
        return [dist.dim for dist in self.distributions]

    @property
    def geometry(self) -> List[Geometry]:
        """ Returns the geometries of the joint distribution. """
        return [dist.geometry for dist in self.distributions]

    def get_parameter_names(self) -> List[str]:
        """ Returns the parameter names of the joint distribution. """
        return [dist.name for dist in self.distributions]

    def get_conditioning_variables(self) -> List[str]:
        """ Return the conditioning variables of the joint distribution. """
        joint_par_names = self.get_parameter_names()
        cond_vars = set()
        for density in self.densities:
            cond_vars.update([par_name for par_name in density.get_parameter_names() if par_name not in joint_par_names])
        return list(cond_vars)

    def get_fixed_variables(self) -> List[str]:
        """ Return the variables that have been conditioned on (fixed). """
        # Extract names of Likelihoods and EvaluatedDensities
        return [density.name for density in self.densities if isinstance(density, Likelihood) or isinstance(density, EvaluatedDensity)]

    def get_density(self, name) -> Density:
        """ Return a density with the given name. """
        for density in self.densities:
            if density.name == name:
                return density
        raise ValueError(f"No density with name {name}.")

    def logd(self, *args, **kwargs):
        """ Evaluate the un-normalized log density function. """

        kwargs = self._parse_args_add_to_kwargs(*args, **kwargs)

        # Check that all parameters are passed by matching parameter names with kwargs keys
        if set(self.get_parameter_names()) != set(kwargs.keys()):
            raise ValueError(f"To evaluate the log density function, all parameters must be passed. Received {kwargs.keys()} and expected {self.get_parameter_names()}.")

        # Evaluate the log density function for each density
        logd = 0
        for density in self.densities:
            logd_kwargs = {key:value for (key,value) in kwargs.items() if key in density.get_parameter_names()}
            logd += density.logd(**logd_kwargs)

        return logd

    def __call__(self, *args, **kwargs) -> JointDistribution:
        """ Condition the joint distribution on the given variables. """
        return self._condition(*args, **kwargs)

    def _condition(self, *args, **kwargs):

        kwargs = self._parse_args_add_to_kwargs(*args, **kwargs)

        # Create new shallow copy of joint density
        new_joint = copy(self) # Shallow copy of self
        new_joint.densities = self.densities[:] # Shallow copy of densities

        # Condition each of the new densities on kwargs relevant to that density
        for i, density in enumerate(new_joint.densities):
            cond_kwargs = {key:value for (key,value) in kwargs.items() if key in density.get_parameter_names()}
            new_joint.densities[i] = density(**cond_kwargs)

        # Hack to reduce the joint distribution to a single density
        # This is useful for current implementation of our samplers
        # but should be removed in the future. Should only be used
        # by the Gibbs sampler.
        if self._allow_reduce:
            return new_joint._reduce_to_single_density()

        return new_joint

    def _parse_args_add_to_kwargs(self, *args, **kwargs):
        """ Parse args and add to kwargs. The args are assumed to follow the order of the parameter names. """
        if len(args)>0:
            ordered_keys = self.get_parameter_names()
            for index, arg in enumerate(args):
                if ordered_keys[index] in kwargs:
                    raise ValueError(f"{ordered_keys[index]} passed as both argument and keyword argument.\nArguments follow the listed parameter names order: {ordered_keys}")
                kwargs[ordered_keys[index]] = arg
        return kwargs

    def _reduce_to_single_density(self):
        """ Reduce the joint distribution to a single density if possible.
        
        This method is a hack to allow our current samplers to work with
        the joint distribution. It should be removed in the future.
        """

        # Count number of distributions and likelihoods
        n_dist = len(self.distributions)
        n_likelihood = len(self.likelihoods)

        # Cant reduce if there are multiple distributions or likelihoods
        if n_dist > 1 or n_likelihood > 1:
            return self
        
        # If exactly one distribution and one likelihood its a Posterior
        if n_dist == 1 and n_likelihood == 1:
            # Ensure parameter names match, otherwise return the joint distribution
            if set(self.likelihoods[0].get_parameter_names()) != set(self.distributions[0].get_parameter_names()):
                return self
            return Posterior(self.likelihoods[0], self.distributions[0])
        
        # If exactly one distribution and no likelihoods its a Distribution
        if n_dist == 1 and n_likelihood == 0:
            return self.distributions[0]

        # If no distributions and exactly one likelihood its a Likelihood
        if n_likelihood == 1 and n_dist == 0:
            return self.likelihoods[0]

    def as_stacked(self) -> StackedJointDistribution:
        """ Return a StackedJointDistribution with the same densities. """
        return StackedJointDistribution(self.densities)

    def __repr__(self):
        msg = f"JointDistribution(\n"
        msg += "    Equation: \n\t"

        # Construct equation expression
        joint_par_names = ",".join(self.get_parameter_names())
        fixed_par_names = ",".join(self.get_fixed_variables())
        if len(joint_par_names) == 0:
            msg += "Constant number"
        else:
            # LHS of equation: p(x,y,z) = or p(x,y|z) ∝
            msg += f"p({joint_par_names}"
            if len(fixed_par_names) > 0:
                msg += f"|{fixed_par_names}) ∝ "
            else:
                msg += ") = "

            # RHS of equation: product of densities
            for density in self.densities:
                par_names = ",".join([density.name])
                cond_vars = ",".join(set(density.get_parameter_names())-set([density.name]))

                # Distributions are written as p(x|y) or p(x). Likelihoods are L(y|x).
                # x=par_names, y=cond_vars.
                if isinstance(density, Likelihood):
                    msg += f"L({cond_vars}|{par_names})"
                elif isinstance(density, Distribution):
                    msg += f"p({par_names}"
                    if len(cond_vars) > 0:
                        msg += f"|{cond_vars}"
                    msg += ")"
        
        msg += "\n"
        msg += "    Densities: \n"

        # Create "Bayesian model" equations
        for density in self.densities:
            msg += f"\t{density.name} ~ {density}\n"

        # Wrap up
        msg += ")"

        return msg


class StackedJointDistribution(JointDistribution, Distribution):
    """ A joint distribution where all parameters are stacked into a single vector.
    
    See :class:`JointDistribution` for more details on the joint distribution.
    """

    @property
    def dim(self):
        return sum(super().dim)

    @property
    def geometry(self):
        return _DefaultGeometry(self.dim)

    def logd(self, stacked_input):
        """ Return the un-normalized log density function stacked joint density. """

        # Split the stacked input into individual inputs and call superclass
        split_indices = np.cumsum(super().dim)  # list(accumulate(super().dim))
        inputs = np.split(stacked_input, split_indices[:-1])
        names = self.get_parameter_names()

        # Create keyword arguments
        kwargs = dict(zip(names, inputs))

        return super().logd(**kwargs)

    def logpdf(self, stacked_input):
        return self.logd(stacked_input)
    
    def _sample(self, Ns=1):
        raise TypeError("StackedJointDistribution does not support sampling.")

    def __repr__(self):
        return "Stacked"+super().__repr__()
