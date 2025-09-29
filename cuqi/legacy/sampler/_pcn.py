import numpy as np
import cuqi
from cuqi.sampler import Sampler

class pCN(Sampler):   
    #Samples target*proposal
    #TODO. Check proposal, needs to be Gaussian and zero mean.
    """Preconditioned Crank-Nicolson sampler 
    
    Parameters
    ----------
    target : `cuqi.distribution.Posterior` or tuple of likelihood and prior objects
        If target is of type cuqi.distribution.Posterior, it represents the posterior distribution.
        If target is a tuple of (cuqi.likelihood.Likelihood, cuqi.distribution.Distribution) objects,
        the first element is considered the likelihood and the second is considered the prior.

    scale : int

    x0 : `np.ndarray` 
      Initial point for the sampler

    callback : callable, *Optional*
        If set this function will be called after every sample.
        The signature of the callback function is `callback(sample, sample_index)`,
        where `sample` is the current sample and `sample_index` is the index of the sample.
        An example is shown in demos/demo31_callback.py.

    Example 
    -------

    This uses a custom logpdf and sample function.

    .. code-block:: python

        # Parameters
        dim = 5 # Dimension of distribution
        mu = np.arange(dim) # Mean of Gaussian
        std = 1 # standard deviation of Gaussian

        # Logpdf function of likelihood
        logpdf_func = lambda x: -1/(std**2)*np.sum((x-mu)**2)

        # sample function of prior N(0,I)
        sample_func = lambda : 0 + 1*np.random.randn(dim,1)

        # Define as UserDefinedDistributions
        likelihood = cuqi.likelihood.UserDefinedLikelihood(dim=dim, logpdf_func=logpdf_func)
        prior = cuqi.distribution.UserDefinedDistribution(dim=dim, sample_func=sample_func)

        # Set up sampler
        sampler = cuqi.sampler.pCN((likelihood,prior), scale = 0.1)

        # Sample
        samples = sampler.sample(5000)

    Example
    -------

    This uses CUQIpy distributions.

    .. code-block:: python

        # Parameters
        dim = 5 # Dimension of distribution
        mu = np.arange(dim) # Mean of Gaussian
        std = 1 # standard deviation of Gaussian

        # Define as UserDefinedDistributions
        model = cuqi.model.Model(lambda x: x, range_geometry=dim, domain_geometry=dim)
        likelihood = cuqi.distribution.Gaussian(mean=model, cov=np.ones(dim)).to_likelihood(mu)
        prior = cuqi.distribution.Gaussian(mean=np.zeros(dim), cov=1)

        target = cuqi.distribution.Posterior(likelihood, prior)

        # Set up sampler
        sampler = cuqi.sampler.pCN(target, scale = 0.1)

        # Sample
        samples = sampler.sample(5000)
        
    """
    def __init__(self, target, scale=None, x0=None, **kwargs):
        super().__init__(target, x0=x0, dim=None, **kwargs) 
        self.scale = scale
    
    @property
    def prior(self):
        if isinstance(self.target, cuqi.distribution.Posterior):
            return self.target.prior
        elif isinstance(self.target,tuple) and len(self.target)==2:
            return self.target[1]

    @property
    def likelihood(self):
        if isinstance(self.target, cuqi.distribution.Posterior):
            return self.target.likelihood
        elif isinstance(self.target,tuple) and len(self.target)==2:
            return self.target[0]


    @Sampler.target.setter 
    def target(self, value):
        if isinstance(value, cuqi.distribution.Posterior):
            self._target = value
            self._loglikelihood = lambda x : self.likelihood.logd(x)
        elif isinstance(value,tuple) and len(value)==2 and \
             (isinstance(value[0], cuqi.likelihood.Likelihood) or isinstance(value[0], cuqi.likelihood.UserDefinedLikelihood))  and \
             isinstance(value[1], cuqi.distribution.Distribution):
            self._target = value
            self._loglikelihood = lambda x : self.likelihood.logd(x)
        else:
            raise ValueError(f"To initialize an object of type {self.__class__}, 'target' need to be of type 'cuqi.distribution.Posterior'.")
        
        #TODO:
        #if not isinstance(self.prior,(cuqi.distribution.Gaussian, cuqi.distribution.Normal)):
        #    raise ValueError("The prior distribution of the target need to be Gaussian")

    @property
    def dim(self):
        if hasattr(self,'target') and hasattr(self.target,'dim'):
            self._dim = self.target.dim
        elif hasattr(self,'target') and isinstance(self.target,tuple) and len(self.target)==2:
            self._dim = self.target[0].dim
        return self._dim

    def _sample(self, N, Nb):
        if self.scale is None:
            raise ValueError("Scale must be set to sample without adaptation. Consider using sample_adapt instead.")

        Ns = N+Nb   # number of simulations

        # allocation
        samples = np.empty((self.dim, Ns))
        loglike_eval = np.empty(Ns)
        acc = np.zeros(Ns, dtype=int)

        # initial state    
        samples[:, 0] = self.x0
        loglike_eval[0] = self._loglikelihood(self.x0)
        acc[0] = 1

        # run MCMC
        for s in range(Ns-1):
            # run component by component
            samples[:, s+1], loglike_eval[s+1], acc[s+1] = self.single_update(samples[:, s], loglike_eval[s])

            self._print_progress(s+2,Ns) #s+2 is the sample number, s+1 is index assuming x0 is the first sample
            self._call_callback(samples[:, s+1], s+1)

        # remove burn-in
        samples = samples[:, Nb:]
        loglike_eval = loglike_eval[Nb:]
        accave = acc[Nb:].mean()   
        print('\nAverage acceptance rate:', accave, '\n')
        #
        return samples, loglike_eval, accave

    def _sample_adapt(self, N, Nb):
        # Set intial scale if not set
        if self.scale is None:
            self.scale = 0.1

        Ns = N+Nb   # number of simulations

        # allocation
        samples = np.empty((self.dim, Ns))
        loglike_eval = np.empty(Ns)
        acc = np.zeros(Ns)

        # initial state    
        samples[:, 0] = self.x0
        loglike_eval[0] = self._loglikelihood(self.x0) 
        acc[0] = 1

        # initial adaptation params 
        Na = int(0.1*N)                              # iterations to adapt
        hat_acc = np.empty(int(np.floor(Ns/Na)))     # average acceptance rate of the chains
        lambd = self.scale
        star_acc = 0.44    # target acceptance rate RW
        i, idx = 0, 0

        # run MCMC
        for s in range(Ns-1):
            # run component by component
            samples[:, s+1], loglike_eval[s+1], acc[s+1] = self.single_update(samples[:, s], loglike_eval[s])
            
            # adapt prop spread using acc of past samples
            if ((s+1) % Na == 0):
                # evaluate average acceptance rate
                hat_acc[i] = np.mean(acc[idx:idx+Na])

                # d. compute new scaling parameter
                zeta = 1/np.sqrt(i+1)   # ensures that the variation of lambda(i) vanishes
                lambd = np.exp(np.log(lambd) + zeta*(hat_acc[i]-star_acc))

                # update parameters
                self.scale = min(lambd, 1)

                # update counters
                i += 1
                idx += Na

            # display iterations
            if ((s+1) % (max(Ns//100,1))) == 0 or (s+1) == Ns-1:
                print("\r",'Sample', s+1, '/', Ns, end="")

            self._call_callback(samples[:, s+1], s+1)

        print("\r",'Sample', s+2, '/', Ns)

        # remove burn-in
        samples = samples[:, Nb:]
        loglike_eval = loglike_eval[Nb:]
        accave = acc[Nb:].mean()   
        print('\nAverage acceptance rate:', accave, 'MCMC scale:', self.scale, '\n')
        
        return samples, loglike_eval, accave

    def single_update(self, x_t, loglike_eval_t):
        # propose state
        xi = self.prior.sample(1).flatten()   # sample from the prior
        x_star = np.sqrt(1-self.scale**2)*x_t + self.scale*xi   # pCN proposal

        # evaluate target
        loglike_eval_star =  self._loglikelihood(x_star) 

        # ratio and acceptance probability
        ratio = loglike_eval_star - loglike_eval_t  # proposal is symmetric
        alpha = min(0, ratio)

        # accept/reject
        u_theta = np.log(np.random.rand())
        if (u_theta <= alpha):
            x_next = x_star
            loglike_eval_next = loglike_eval_star
            acc = 1
        else:
            x_next = x_t
            loglike_eval_next = loglike_eval_t
            acc = 0
        
        return x_next, loglike_eval_next, acc
    
    