import cuqi
import inspect
import numpy as np

# This import makes suggest_sampler easier to read
import cuqi.experimental.mcmc as samplers 


class SamplerRecommender(object):

    def __init__(self, target:cuqi.density.Density, exceptions = []):

        self._target = target
        self.exceptions = exceptions

        self._create_ordering()

    @property
    def target(self) -> cuqi.density.Density:
        """ Return the target density. """
        return self._target
    
    @target.setter
    def target(self, value:cuqi.density.Density):
        """ Set the target density. Runs validation of the target. """
        if value is None:
            raise ValueError("Target needs to be of type cuqi.density.Density.")
        self._target = value

    def _create_ordering(self):
        self._ordering = [
            # Direct and Conjugate samplers
            (samplers.Direct, {}),
            (samplers.Conjugate, {}),
            (samplers.ConjugateApprox, {}),
            # Specialized samplers
            (samplers.LinearRTO, {}),
            (samplers.RegularizedLinearRTO, {}),
            (samplers.UGLA, {}),
            # Gradient.based samplers (Hamiltonian and Langevin)
            (samplers.NUTS, {}),
            (samplers.MALA, {}),
            (samplers.ULA, {}),
            # Gibbs and Componentwise samplers
            (samplers.HybridGibbs, {"sampling_strategy" : self.recommend_HybridGibbs_sampling_strategy(as_string = False)}),
            (samplers.CWMH, {"scale" : 0.05*np.ones(self._target.dim),
                            "x0" : 0.5*np.ones(self._target.dim)}),
            # Proposal based samplers
            (samplers.PCN, {"scale" : 0.02}),
            (samplers.MH, {}),
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

        # TODO: Conditions for suggestions, e.g., only use CWMH when the dimension of the problem is low

        valid_samplers = self.valid_samplers(as_string = False)

        for suggestion, values in self._ordering:
            if suggestion in valid_samplers and suggestion not in self.exceptions:
                # Sampler found
                if as_string:
                    return suggestion.__name__
                else:
                    return suggestion(self.target, **values)

        # No sampler can be suggested
        return None

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

            recommender = SamplerRecommender(conditional)
            sampler = recommender.recommend(as_string = as_string)

            if sampler is None:
                return None
            
            suggested_samplers[par_name] = sampler

        return suggested_samplers