import cuqi
import inspect
import numpy as np

# This import makes suggest_sampler easier to read
import cuqi.experimental.mcmc as samplers 


class SamplerRecommender(object):
    """
    This class can be used to automatically choose a sampler.

    Parameters
    ----------
    target: Density or JointDistribution
        Distribution to get sampler recommendations for.

    exceptions: list[cuqi.experimental.mcmc.Sampler], *optional*
        Samplers not to be recommended.
    
    Example
    -------
    .. code-block:: python
        import numpy as np
        from cuqi.distribution import Gamma, Gaussian, JointDistribution
        from cuqi.experimental import SamplerRecommender

        x = Gamma(1, 1)
        y = Gaussian(np.zeros(2), cov=lambda x: 1 / x)
        target = JointDistribution(y, x)(y=1)

        recommender = SamplerRecommender(target)
        valid_samplers = recommender.valid_samplers()
        recommended_sampler = recommender.recommend()
        print("Valid samplers:", valid_samplers)
        print("Recommended sampler:\n", recommended_sampler)

    """

    def __init__(self, target:cuqi.density.Density, exceptions = []):
        self._target = target
        self._exceptions = exceptions
        self._create_ordering()

    @property
    def target(self) -> cuqi.density.Density:
        """ Return the target Distribution. """
        return self._target

    @target.setter
    def target(self, value:cuqi.density.Density):
        """ Set the target Distribution. Runs validation of the target. """
        if value is None:
            raise ValueError("Target needs to be of type cuqi.density.Density.")
        self._target = value

    def _create_ordering(self):
        """
        Every element in the ordering consists of a tuple:
        (
            Sampler: Class
            boolean: additional conditions on the target
            parameters: additional parameters to be passed to the sampler once initialized
        )
        """
        number_of_components = np.sum(self._target.dim)

        self._ordering = [
            # Direct and Conjugate samplers
            (samplers.Direct, True, {}),
            (samplers.Conjugate, True, {}),
            (samplers.ConjugateApprox, True, {}),
            # Specialized samplers
            (samplers.LinearRTO, True, {}),
            (samplers.RegularizedLinearRTO, True, {}),
            (samplers.UGLA, True, {}),
            # Gradient.based samplers (Hamiltonian and Langevin)
            (samplers.NUTS, True, {}),
            (samplers.MALA, True, {}),
            (samplers.ULA, True, {}),
            # Gibbs and Componentwise samplers
            (samplers.HybridGibbs, True, {"sampling_strategy" : self.recommend_HybridGibbs_sampling_strategy(as_string = False)}),
            (samplers.CWMH, number_of_components <= 100, {"scale" : 0.05*np.ones(number_of_components),
                            "initial_point" : 0.5*np.ones(number_of_components)}),
            # Proposal based samplers
            (samplers.PCN, True, {"scale" : 0.02}),
            (samplers.MH, number_of_components <= 1000, {}),
        ]

    @property
    def ordering(self):
        """ Returns the ordered list of recommendation rules used by the recommender. """
        return self._ordering

    def valid_samplers(self, as_string = True):
        """
        Finds all possible samplers that can be used for sampling from the target distribution.

        Parameters
        ----------

        as_string : boolean
            Whether to return the name of the sampler as a string instead of instantiating a sampler. *Optional*

        """

        all_samplers = [(name, cls) for name, cls in inspect.getmembers(cuqi.experimental.mcmc, inspect.isclass) if issubclass(cls, cuqi.experimental.mcmc.Sampler)]
        valid_samplers = []

        for name, sampler in all_samplers:
            try:
                sampler(self.target)
                valid_samplers += [name if as_string else sampler]
            except:
                pass

        # Need a separate case for HybridGibbs
        if self.valid_HybridGibbs_sampling_strategy() is not None:
            valid_samplers += [cuqi.experimental.mcmc.HybridGibbs.__name__ if as_string else cuqi.experimental.mcmc.HybridGibbs]

        return valid_samplers

    def valid_HybridGibbs_sampling_strategy(self, as_string = True):
        """
            Find all possible sampling strategies to be used with the HybridGibbs sampler.
            Returns None if no sampler could be suggested for at least one conditional distribution.

            Parameters
            ----------

            as_string : boolean
                Whether to return the name of the samplers in the sampling strategy as a string instead of instantiating samplers. *Optional*
        

        """

        if not isinstance(self.target, cuqi.distribution.JointDistribution):
            return None

        par_names = self.target.get_parameter_names()

        valid_samplers = dict()
        for par_name in par_names:
            conditional_params = {par_name_: np.ones(self.target.dim[i]) for i, par_name_ in enumerate(par_names) if par_name_ != par_name}
            conditional = self.target(**conditional_params)

            recommender = SamplerRecommender(conditional)
            samplers = recommender.valid_samplers(as_string)
            if len(samplers) == 0:
                return None

            valid_samplers[par_name] = samplers

        return valid_samplers

    def recommend(self, as_string = False):
        """
            Suggests a possible sampler that can be used for sampling from the target distribution.
            Return None if no sampler could be suggested.

            Parameters
            ----------

            as_string : boolean
                Whether to return the name of the sampler as a string instead of instantiating a sampler. *Optional*

        """

        valid_samplers = self.valid_samplers(as_string = False)

        for suggestion, flag, values in self._ordering:
            if flag and (suggestion in valid_samplers) and (suggestion not in self._exceptions):
                # Sampler found
                if as_string:
                    return suggestion.__name__
                else:
                    return suggestion(self.target, **values)

        # No sampler can be suggested
        raise ValueError("Cannot suggest any sampler. Either the provided distribution is incorrectly defined or there are too many exceptions provided.")

    def recommend_HybridGibbs_sampling_strategy(self, as_string = False):
        """
            Suggests a possible sampling strategy to be used with the HybridGibbs sampler.
            Returns None if no sampler could be suggested for at least one conditional distribution.

            Parameters
            ----------

            target : `cuqi.distribution.JointDistribution`
                The target distribution get a sampling strategy for.
            
            as_string : boolean
                Whether to return the name of the samplers in the sampling strategy as a string instead of instantiating samplers. *Optional*
        
        """

        if not isinstance(self.target, cuqi.distribution.JointDistribution):
            return None

        par_names = self.target.get_parameter_names()

        suggested_samplers = dict()
        for par_name in par_names:
            conditional_params = {par_name_: np.ones(self.target.dim[i]) for i, par_name_ in enumerate(par_names) if par_name_ != par_name}
            conditional = self.target(**conditional_params)

            recommender = SamplerRecommender(conditional, exceptions = self._exceptions.copy())
            sampler = recommender.recommend(as_string = as_string)

            if sampler is None:
                return None

            suggested_samplers[par_name] = sampler

        return suggested_samplers
