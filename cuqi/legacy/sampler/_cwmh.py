import numpy as np
import cuqi
from cuqi.sampler import ProposalBasedSampler


class CWMH(ProposalBasedSampler):
    """Component-wise Metropolis Hastings sampler.

    Allows sampling of a target distribution by a component-wise random-walk sampling of a proposal distribution along with an accept/reject step.

    Parameters
    ----------

    target : `cuqi.distribution.Distribution` or lambda function
        The target distribution to sample. Custom logpdfs are supported by using a :class:`cuqi.distribution.UserDefinedDistribution`.
    
    proposal : `cuqi.distribution.Distribution` or callable method
        The proposal to sample from. If a callable method it should provide a single independent sample from proposal distribution. Defaults to a Gaussian proposal.  *Optional*.

    scale : float
        Scale parameter used to define correlation between previous and proposed sample in random-walk.  *Optional*.

    x0 : ndarray
        Initial parameters. *Optional*

    dim : int
        Dimension of parameter space. Required if target and proposal are callable functions. *Optional*.

    callback : callable, *Optional*
        If set this function will be called after every sample.
        The signature of the callback function is `callback(sample, sample_index)`,
        where `sample` is the current sample and `sample_index` is the index of the sample.
        An example is shown in demos/demo31_callback.py.

    Example
    -------
    .. code-block:: python

        # Parameters
        dim = 5 # Dimension of distribution
        mu = np.arange(dim) # Mean of Gaussian
        std = 1 # standard deviation of Gaussian

        # Logpdf function
        logpdf_func = lambda x: -1/(std**2)*np.sum((x-mu)**2)

        # Define distribution from logpdf as UserDefinedDistribution (sample and gradients also supported as inputs to UserDefinedDistribution)
        target = cuqi.distribution.UserDefinedDistribution(dim=dim, logpdf_func=logpdf_func)

        # Set up sampler
        sampler = cuqi.sampler.CWMH(target, scale=1)

        # Sample
        samples = sampler.sample(2000)

    """
    def __init__(self, target,  proposal=None, scale=1, x0=None, dim = None, **kwargs):
        super().__init__(target, proposal=proposal, scale=scale,  x0=x0, dim=dim, **kwargs)
        
    @ProposalBasedSampler.proposal.setter 
    def proposal(self, value):
        fail_msg = "Proposal should be either None, cuqi.distribution.Distribution conditioned only on 'location' and 'scale', lambda function, or cuqi.distribution.Normal conditioned only on 'mean' and 'std'"

        if value is None:
            self._proposal = cuqi.distribution.Normal(mean = lambda location:location,std = lambda scale:scale, geometry=self.dim)

        elif isinstance(value, cuqi.distribution.Distribution) and sorted(value.get_conditioning_variables())==['location','scale']:
            self._proposal = value

        elif isinstance(value, cuqi.distribution.Normal) and sorted(value.get_conditioning_variables())==['mean','std']:
            self._proposal = value(mean = lambda location:location, std = lambda scale:scale)

        elif not isinstance(value, cuqi.distribution.Distribution) and callable(value):
            self._proposal = value

        else:
            raise ValueError(fail_msg)


    def _sample(self, N, Nb):
        Ns = N+Nb   # number of simulations

        # allocation
        samples = np.empty((self.dim, Ns))
        target_eval = np.empty(Ns)
        acc = np.zeros((self.dim, Ns), dtype=int)

        # initial state    
        samples[:, 0] = self.x0
        target_eval[0] = self.target.logd(self.x0)
        acc[:, 0] = np.ones(self.dim)

        # run MCMC
        for s in range(Ns-1):
            # run component by component
            samples[:, s+1], target_eval[s+1], acc[:, s+1] = self.single_update(samples[:, s], target_eval[s])

            self._print_progress(s+2,Ns) #s+2 is the sample number, s+1 is index assuming x0 is the first sample
            self._call_callback(samples[:, s+1], s+1)

        # remove burn-in
        samples = samples[:, Nb:]
        target_eval = target_eval[Nb:]
        acccomp = acc[:, Nb:].mean(axis=1)   
        print('\nAverage acceptance rate all components:', acccomp.mean(), '\n')
        
        return samples, target_eval, acccomp

    def _sample_adapt(self, N, Nb):
        # this follows the vanishing adaptation Algorithm 4 in:
        # Andrieu and Thoms (2008) - A tutorial on adaptive MCMC
        Ns = N+Nb   # number of simulations

        # allocation
        samples = np.empty((self.dim, Ns))
        target_eval = np.empty(Ns)
        acc = np.zeros((self.dim, Ns), dtype=int)

        # initial state
        samples[:, 0] = self.x0
        target_eval[0] = self.target.logd(self.x0)
        acc[:, 0] = np.ones(self.dim)

        # initial adaptation params 
        Na = int(0.1*N)                                        # iterations to adapt
        hat_acc = np.empty((self.dim, int(np.floor(Ns/Na))))     # average acceptance rate of the chains
        lambd = np.empty((self.dim, int(np.floor(Ns/Na)+1)))     # scaling parameter \in (0,1)
        lambd[:, 0] = self.scale
        star_acc = 0.21/self.dim + 0.23    # target acceptance rate RW
        i, idx = 0, 0

        # run MCMC
        for s in range(Ns-1):
            # run component by component
            samples[:, s+1], target_eval[s+1], acc[:, s+1] = self.single_update(samples[:, s], target_eval[s])
            
            # adapt prop spread of each component using acc of past samples
            if ((s+1) % Na == 0):
                # evaluate average acceptance rate
                hat_acc[:, i] = np.mean(acc[:, idx:idx+Na], axis=1)

                # compute new scaling parameter
                zeta = 1/np.sqrt(i+1)   # ensures that the variation of lambda(i) vanishes
                lambd[:, i+1] = np.exp(np.log(lambd[:, i]) + zeta*(hat_acc[:, i]-star_acc))  

                # update parameters
                self.scale = np.minimum(lambd[:, i+1], np.ones(self.dim))

                # update counters
                i += 1
                idx += Na

            # display iterations 
            self._print_progress(s+2,Ns) #s+2 is the sample number, s+1 is index assuming x0 is the first sample
            self._call_callback(samples[:, s+1], s+1)
            
        # remove burn-in
        samples = samples[:, Nb:]
        target_eval = target_eval[Nb:]
        acccomp = acc[:, Nb:].mean(axis=1)
        print('\nAverage acceptance rate all components:', acccomp.mean(), '\n')
        
        return samples, target_eval, acccomp

    def single_update(self, x_t, target_eval_t):
        if isinstance(self.proposal,cuqi.distribution.Distribution):
            x_i_star = self.proposal(location= x_t, scale = self.scale).sample()
        else:
            x_i_star = self.proposal(x_t, self.scale) 
        x_star = x_t.copy()
        acc = np.zeros(self.dim)

        for j in range(self.dim):
            # propose state
            x_star[j] = x_i_star[j]

            # evaluate target
            target_eval_star = self.target.logd(x_star)

            # ratio and acceptance probability
            ratio = target_eval_star - target_eval_t  # proposal is symmetric
            alpha = min(0, ratio)

            # accept/reject
            u_theta = np.log(np.random.rand())
            if (u_theta <= alpha):
                x_t[j] = x_i_star[j]
                target_eval_t = target_eval_star
                acc[j] = 1
            else:
                pass
                # x_t[j]       = x_t[j]
                # target_eval_t = target_eval_t
            x_star = x_t.copy()
        #
        return x_t, target_eval_t, acc
