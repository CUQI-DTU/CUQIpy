import scipy as sp
import scipy.stats as sps
import numpy as np

from cuqi.distribution import Laplace_diff
# import matplotlib
# import matplotlib.pyplot as plt
eps = np.finfo(float).eps

import cuqi
from cuqi.solver import CGLS
from cuqi.samples import Samples
from abc import ABC, abstractmethod

#===================================================================
class Sampler(ABC):

    def __init__(self, target, x0=None, dim=None):

        self._dim = dim
        if hasattr(target,'dim'): 
            if self._dim is None:
                self._dim = target.dim 
            elif self._dim != target.dim:
                raise ValueError("'dim' need to be None or equal to 'target.dim'") 
        elif x0 is not None:
            self._dim = len(x0)

        self.target = target

        if x0 is None:
            x0 = np.ones(self.dim)
        self.x0 = x0

    @property
    def geometry(self):
        if hasattr(self, 'target') and hasattr(self.target, 'geometry'):
            geom =  self.target.geometry
        else:
            geom = cuqi.geometry._DefaultGeometry(self.dim)
        return geom

    @property 
    def target(self):
        return self._target 

    @target.setter 
    def target(self, value):
        if  not isinstance(value, cuqi.distribution.Distribution) and callable(value):
            # obtain self.dim
            if self.dim is not None:
                dim = self.dim
            else:
                raise ValueError(f"If 'target' is a lambda function, the parameter 'dim' need to be specified when initializing {self.__class__}.")

            # set target
            self._target = cuqi.distribution.UserDefinedDistribution(logpdf_func=value, dim = dim)

        elif isinstance(value, cuqi.distribution.Distribution):
            self._target = value
        else:
            raise ValueError("'target' need to be either a lambda function or of type 'cuqi.distribution.Distribution'")


    @property
    def dim(self):
        if hasattr(self,'target') and hasattr(self.target,'dim'):
            self._dim = self.target.dim 
        return self._dim
    

    def sample(self,N,Nb=0):
        # Get samples from the samplers sample method
        result = self._sample(N,Nb)
        return self._create_Sample_object(result,N+Nb)

    def sample_adapt(self,N,Nb=0):
        # Get samples from the samplers sample method
        result = self._sample_adapt(N,Nb)
        return self._create_Sample_object(result,N+Nb)

    def _create_Sample_object(self,result,N):
        loglike_eval = None
        acc_rate = None
        if isinstance(result,tuple):
            #Unpack samples+loglike+acc_rate
            s = result[0]
            if len(result)>1: loglike_eval = result[1]
            if len(result)>2: acc_rate = result[2]
            if len(result)>3: raise TypeError("Expected tuple of at most 3 elements from sampling method.")
        else:
            s = result
                
        #Store samples in cuqi samples object if more than 1 sample
        if N==1:
            if len(s) == 1 and isinstance(s,np.ndarray): #Extract single value from numpy array
                s = s.ravel()[0]
            else:
                s = s.flatten()
        else:
            s = Samples(s, self.geometry)#, geometry = self.geometry)
            s.loglike_eval = loglike_eval
            s.acc_rate = acc_rate
        return s

    @abstractmethod
    def _sample(self,N,Nb):
        pass

    @abstractmethod
    def _sample_adapt(self,N,Nb):
        pass

    def _print_progress(self,s,Ns):
        """Prints sampling progress"""
        if (s % (max(Ns//100,1))) == 0:
            print("\r",'Sample', s, '/', Ns, end="")
        elif s == Ns:
            print("\r",'Sample', s, '/', Ns)

class ProposalBasedSampler(Sampler,ABC):
    def __init__(self, target,  proposal=None, scale=1, x0=None, dim=None):
        #TODO: after fixing None dim
        #if dim is None and hasattr(proposal,'dim'):
        #    dim = proposal.dim
        super().__init__(target, x0=x0, dim=dim)

        self.proposal =proposal
        self.scale = scale


    @property 
    def proposal(self):
        return self._proposal 

    @proposal.setter 
    def proposal(self, value):
        self._proposal = value

    @property
    def geometry(self):
        geom1, geom2 = None, None
        if hasattr(self, 'proposal') and hasattr(self.proposal, 'geometry') and self.proposal.geometry.dim is not None:
            geom1=  self.proposal.geometry
        if hasattr(self, 'target') and hasattr(self.target, 'geometry') and self.target.geometry.dim is not None:
            geom2 = self.target.geometry

        if not isinstance(geom1,cuqi.geometry._DefaultGeometry):
            return geom1
        elif not isinstance(geom2,cuqi.geometry._DefaultGeometry): 
            return geom2
        else:
            return cuqi.geometry._DefaultGeometry(self.dim)



# another implementation is in https://github.com/mfouesneau/NUTS
class NUTS(Sampler):
    """No-U-Turn Sampler (Hoffman and Gelman, 2014).

    Samples a distribution given its logpdf and gradient using a Hamiltonian Monte Carlo (HMC) algorithm with automatic parameter tuning.

    For more details see: See Hoffman, M. D., & Gelman, A. (2014). The no-U-turn sampler: Adaptively setting path lengths in Hamiltonian Monte Carlo. Journal of Machine Learning Research, 15, 1593-1623.

    Parameters
    ----------

    target : `cuqi.distribution.Distribution`
        The target distribution to sample. Must have logpdf and gradient method. Custom logpdfs and gradients are supported by using a :class:`cuqi.distribution.UserDefinedDistribution`.
    
    x0 : ndarray
        Initial parameters. *Optional*

    dim : int
        Dimension of parameter space. Required if target logpdf and gradient are callable functions. *Optional*.

    Example
    -------
    .. code-block:: python

        # Parameters
        dim = 5 # Dimension of distribution
        mu = np.arange(dim) # Mean of Gaussian
        std = 1 # standard deviation of Gaussian

        # Logpdf function
        logpdf_func = lambda x: -1/(std**2)*np.sum((x-mu)**2)
        gradient_func = lambda x: -1/(std**2)*(x - mu)

        # Define distribution from logpdf as UserDefinedDistribution (sample and gradients also supported)
        target = cuqi.distribution.UserDefinedDistribution(dim=dim, logpdf_func=logpdf_func, gradient_func=gradient_func)

        # Set up sampler
        sampler = cuqi.sampler.NUTS(target)

        # Sample
        samples = sampler.sample(2000)

    """
    def __init__(self, target, x0=None, dim=None, maxdepth=20):
        super().__init__(target, x0=x0, dim=dim)
        self.maxdepth = maxdepth


    def potential(self,x):
        """Potential of likelihood+prior. Also returns the gradient"""
        logpdf = -self.target.logpdf(x)
        grad = -self.target.gradient(x)
        return logpdf, grad

    def _sample_adapt(self, N, Nb):
        return self._sample(N,Nb)

    def _sample(self, N, Nb):

        # Allocation
        Ns = Nb+N               # total number of chains
        theta = np.empty((self.dim, Ns))
        pot_eval = np.empty(Ns)

        # Initial state
        theta[:, 0] = self.x0
        pot_eval[0], grad_pot = self.potential(self.x0)
        
        # Init parameters with dual averaging
        epsilon = self._FindGoodEpsilon(theta[:, 0], pot_eval[0], grad_pot)
        mu = np.log(10*epsilon)
        gamma, t_0, kappa = 0.05, 10, 0.75
        epsilon_bar, H_bar = 1, 0
        delta = 0.65  # per stan: https://mc-stan.org/docs/2_18/reference-manual/hmc-algorithm-parameters.html

        # run NUTS
        for k in range(1, Ns):        
            q_k = theta[:, k-1]                            # initial position (parameters)
            p_k = self._Kfun(1, 'sample')                  # resample momentum vector
            H = -pot_eval[k-1] - self._Kfun(p_k, 'eval')   # Hamiltonian

            # slice variable
            log_u = H - np.random.exponential(1)
            # u = np.random.uniform(0, np.exp(H))

            # if NUTS does not move, the next sample will be the previous one
            theta[:, k] = q_k
            pot_eval[k] = pot_eval[k-1]

            # run NUTS
            j, s, n = 0, 1, 1
            q_minus, q_plus = np.copy(q_k), np.copy(q_k)
            p_minus, p_plus = np.copy(p_k), np.copy(p_k)
            grad_pot_minus, grad_pot_plus = np.copy(grad_pot), np.copy(grad_pot)
            while (s == 1) and (j <= self.maxdepth):
                # sample a direction
                v = int(2*(np.random.rand() < 0.5)-1)

                # build tree: doubling procedure
                if (v == -1):
                    q_minus, p_minus, grad_pot_minus, _, _, _, q_p, pot_p, grad_pot_p, n_p, s_p, alpha, n_alpha = \
                        self._BuildTree(q_minus, p_minus, grad_pot_minus, H, log_u, v, j, epsilon)
                else:
                    _, _, _, q_plus, p_plus, grad_pot_plus, q_p, pot_p, grad_pot_p, n_p, s_p, alpha, n_alpha = \
                        self._BuildTree(q_plus, p_plus, grad_pot_plus, H, log_u, v, j, epsilon)

                # Metropolis step
                alpha = min(1,n_p/n) #min(0, np.log(n_p) - np.log(n)) TODO...
                if (s_p == 1) and ((np.random.rand()) <= alpha) and (np.isnan(pot_p) == False): #TODO. Removed np.log
                    theta[:, k] = q_p
                    pot_eval[k] = pot_p
                    grad_pot = np.copy(grad_pot_p)

                # update number of particles, tree level, and stopping criterion
                n += n_p
                j += 1
                s = s_p*int(((q_plus-q_minus)@p_minus)>=0)*int(((q_plus-q_minus)@p_plus)>=0)

            # adapt epsilon during burn-in using dual averaging
            if (k < Nb):
                eta1 = 1/(k + t_0)
                H_bar = (1-eta1)*H_bar + eta1*(delta - (alpha/n_alpha))
                #
                epsilon = np.exp(mu - (np.sqrt(k)/gamma)*H_bar)
                # print('\n', k, '\t', epsilon)
                eta = k**(-kappa)
                epsilon_bar = np.exp((1-eta)*np.log(epsilon_bar) + eta*np.log(epsilon))
            elif (k == Nb):
                epsilon = epsilon_bar   # fix epsilon after burn-in
                
            self._print_progress(k+1,Ns) #k+1 is the sample number, k is index assuming x0 is the first sample

            # msg
            if (np.mod(k, 25) == 0):
                if np.isnan(pot_eval[k]):
                    raise NameError('NaN potential func')

        # apply burn-in 
        theta = theta[:, Nb:]
        pot_eval = pot_eval[Nb:]

        return theta, pot_eval, epsilon

    # auxiliary standard Gaussian PDF: kinetic energy function
    # d_log_2pi = d*np.log(2*np.pi)
    def _Kfun(self,p, flag):
        if flag == 'eval': # evaluate
            return 0.5*( (p.T @ p) ) #+ d_log_2pi 
        if flag == 'sample': # sample
            return np.random.normal(size=self.dim)

    def _FindGoodEpsilon(self,theta, pot, grad_pot):
        epsilon = 1
        r = self._Kfun(1, 'sample')
        H = -pot - self._Kfun(r, 'eval')
        _, r_p, pot_p, grad_pot_p = self._Leapfrog(theta, r, grad_pot, epsilon)

        # additional step to correct in case of inf values
        k = 1
        while np.isinf(pot_p) or np.isinf(grad_pot_p).any():
            k *= 0.5
            _, r_p, pot_p, _ = self._Leapfrog(theta, r, grad_pot, epsilon*k)
        epsilon = 0.5*k*epsilon

        # doubles/halves the value of epsilon until the accprob of the Langevin proposal crosses 0.5
        H_p = -pot_p - self._Kfun(r_p, 'eval')
        log_ratio = H_p - H
        a = 1 if log_ratio > np.log(0.5) else -1
        while (a*log_ratio > -a*np.log(2)):
            epsilon = (2**a)*epsilon
            _, r_p, pot_p, _ = self._Leapfrog(theta, r, grad_pot, epsilon)
            H_p = -pot_p - self._Kfun(r_p, 'eval')
            log_ratio = H_p - H

        return epsilon

    #=========================================================================
    def _Leapfrog(self,theta_old, r_old, grad_pot_old, epsilon):
        # symplectic integrator: trajectories preserve phase space volumen
        r_new = r_old - (epsilon/2)*grad_pot_old       # half-step
        theta_new = theta_old + epsilon*r_new          # full-step
        pot_new, grad_pot_new = self.potential(theta_new)   # new gradient
        r_new -= (epsilon/2)*grad_pot_new              # half-step

        return theta_new, r_new, pot_new, grad_pot_new


    #=========================================================================
    # @functools.lru_cache(maxsize=128)
    def _BuildTree(self, theta, r, grad_pot, H, log_u, v, j, epsilon, Delta_max=1000):
        if (j == 0): 
            # single leapfrog step in the direction v
            theta_p, r_p, pot_p, grad_pot_p = self._Leapfrog(theta, r, grad_pot, v*epsilon)
            #
            H_p = -pot_p - self._Kfun(r_p, 'eval')     # Hamiltonian eval
            n_p = int(log_u <= H_p)              # if particle is in the slice
            s_p = int((log_u-Delta_max) < H_p)   # check U-turn

            #TODO: Quick fix to avoid overflow
            diff_H = H_p-H
            if diff_H>100:
                alpha_p = 1
            else:
                alpha_p = min(1, np.exp(diff_H))    # logalpha_p = min(0, H_p - H)

            return theta_p, r_p, grad_pot_p, theta_p, r_p, grad_pot_p, theta_p, pot_p, grad_pot_p, n_p, s_p, alpha_p, 1
            
        else: 
            # recursion: build the left/right subtrees
            theta_minus, r_minus, grad_pot_minus, theta_plus, r_plus, grad_pot_plus, \
            theta_p, pot_p, grad_pot_p, n_p, s_p, alpha_p, n_alpha_p = \
                self._BuildTree(theta, r, grad_pot, H, log_u, v, j-1, epsilon)
            if (s_p == 1): # do only if the stopping criteria does not verify at the first subtree
                if (v == -1):
                    theta_minus, r_minus, grad_pot_minus, _, _, _, \
                    theta_pp, pot_pp, grad_pot_pp, n_pp, s_pp, alpha_pp, n_alpha_pp = \
                        self._BuildTree(theta_minus, r_minus, grad_pot_minus, H, log_u, v, j-1, epsilon)
                else:
                    _, _, _, theta_plus, r_plus, grad_pot_plus, \
                    theta_pp, pot_pp, grad_pot_pp, n_pp, s_pp, alpha_pp, n_alpha_pp = \
                        self._BuildTree(theta_plus, r_plus, grad_pot_plus, H, log_u, v, j-1, epsilon)

                # Metropolis step
                #TODO: Check this update. Use log instead?
                alpha2 = n_pp/max(1, n_p+n_pp) #np.log(n_pp) - np.log(n_p+n_pp)
                if ((np.random.rand()) <= alpha2):
                    theta_p = np.copy(theta_pp)
                    pot_p = np.copy(pot_pp)
                    grad_pot_p = np.copy(grad_pot_pp)

                # update number of particles and stopping criterion
                alpha_p += alpha_pp
                n_alpha_p += n_alpha_pp
                n_p += n_pp   
                s_p = s_pp*(((theta_plus-theta_minus)@r_minus)>=0)*(((theta_plus-theta_minus)@r_plus)>=0)

            return theta_minus, r_minus, grad_pot_minus, theta_plus, r_plus, grad_pot_plus, theta_p, pot_p, grad_pot_p, n_p, s_p, alpha_p, n_alpha_p


class Linear_RTO(Sampler):
    """
    Linear RTO (Randomize-Then-Optimize) sampler.

    Samples posterior related to the inverse problem with Gaussian likelihood and prior, and where the forward model is Linear.

    Parameters
    ------------
    target : `cuqi.distribution.Posterior` or 5-dimensional tuple.
        If target is of type cuqi.distribution.Posterior, it represents the posterior distribution.
        If target is a 5-dimensional tuple, it assumes the following structure:
        (data, model, L_sqrtprec, P_mean, P_sqrtrec)
        
        Here:
        data: is a m-dimensional numpy array containing the measured data.
        model: is a m by n dimensional matrix or LinearModel representing the forward model.
        L_sqrtprec: is the squareroot of the precision matrix of the Gaussian likelihood.
        P_mean: is the prior mean.
        P_sqrtprec: is the squareroot of the precision matrix of the Gaussian mean.

    x0 : `np.ndarray` 
        Initial point for the sampler. *Optional*.

    maxit : int
        Maximum number of iterations of the inner CGLS solver. *Optional*.

    tol : float
        Tolerance of the inner CGLS solver. *Optional*.
        
    """
    def __init__(self, target, x0=None, maxit=10, tol=1e-6, shift=0):
        
        # Accept tuple of inputs and construct posterior
        if isinstance(target, tuple) and len(target) == 5:
            # Structure (data, model, L_sqrtprec, P_mean, P_sqrtprec)
            data = target[0]
            model = target[1]
            L_sqrtprec = target[2]
            P_mean = target[3]
            P_sqrtprec = target[4]


            # If numpy matrix convert to CUQI model
            if isinstance(model, np.ndarray) and len(model.shape) == 2:
                model = cuqi.model.LinearModel(model)

            # Check model input
            if not isinstance(model, cuqi.model.LinearModel):
                raise TypeError("Model needs to be cuqi.model.LinearModel or matrix")

            # Likelihood
            L = cuqi.distribution.GaussianSqrtPrec(model, L_sqrtprec).to_likelihood(data)

            # Prior TODO: allow multiple priors stacked
            #if isinstance(P_mean, list) and isinstance(P_sqrtprec, list):
            #    P = cuqi.distribution.JointGaussianSqrtPrec(P_mean, P_sqrtprec)
            #else:
            P = cuqi.distribution.GaussianSqrtPrec(P_mean, P_sqrtprec)

            # Construct posterior
            target = cuqi.distribution.Posterior(L, P)

        super().__init__(target, x0=x0)

        # Check target type
        if not isinstance(target, cuqi.distribution.Posterior):
            raise ValueError(f"To initialize an object of type {self.__class__}, 'target' need to be of type 'cuqi.distribution.Posterior'.")       

        # Check Linear model and Gaussian prior+likelihood
        if not isinstance(self.model, cuqi.model.LinearModel):
            raise TypeError("Model needs to be linear")

        if not hasattr(self.likelihood.distribution, "sqrtprec"):
            raise TypeError("Distribution in Likelihood must contain a sqrtprec attribute")

        if not hasattr(self.prior, "sqrtprec"):
            raise TypeError("prior must contain a sqrtprec attribute")

        if not hasattr(self.prior, "sqrtprecTimesMean"):
            raise TypeError("Prior must contain a sqrtprecTimesMean attribute")

        # Modify initial guess        
        if x0 is not None:
            self.x0 = x0
        else:
            self.x0 = np.zeros(self.prior.dim)

        # Other parameters
        self.maxit = maxit
        self.tol = tol        
        self.shift = 0
                
        L1 = self.likelihood.distribution.sqrtprec
        L2 = self.prior.sqrtprec
        L2mu = self.prior.sqrtprecTimesMean

        # pre-computations
        self.m = len(self.data)
        self.n = len(self.x0)
        self.b_tild = np.hstack([L1@self.data, L2mu]) 

        if not callable(self.model):
            self.M = sp.sparse.vstack([L1@self.model, L2])
        else:
            # in this case, model is a function doing forward and backward operations
            def M(x, flag):
                if flag == 1:
                    out1 = L1 @ self.model.forward(x)
                    out2 = L2 @ x
                    out  = np.hstack([out1, out2])
                elif flag == 2:
                    idx = int(self.m)
                    out1 = self.model.adjoint(L1.T@x[:idx])
                    out2 = L2.T @ x[idx:]
                    out  = out1 + out2                
                return out   
            self.M = M       

    @property
    def prior(self):
        return self.target.prior

    @property
    def likelihood(self):
        return self.target.likelihood

    @property
    def model(self):
        return self.target.model     
    
    @property
    def data(self):
        return self.target.data

    def _sample(self, N, Nb):   
        Ns = N+Nb   # number of simulations        
        samples = np.empty((self.n, Ns))
                     
        # initial state   
        samples[:, 0] = self.x0
        for s in range(Ns-1):
            y = self.b_tild + np.random.randn(len(self.b_tild))
            sim = CGLS(self.M, y, samples[:, s], self.maxit, self.tol, self.shift)            
            samples[:, s+1], _ = sim.solve()

            self._print_progress(s+2,Ns) #s+2 is the sample number, s+1 is index assuming x0 is the first sample
        
        # remove burn-in
        samples = samples[:, Nb:]
        
        return samples, None, None

    def _sample_adapt(self, N, Nb):
        return self._sample(N,Nb)


#===================================================================
#===================================================================
#===================================================================
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
    def __init__(self, target,  proposal=None, scale=1, x0=None, dim = None):
        super().__init__(target, proposal=proposal, scale=scale,  x0=x0, dim=dim)
        
    @ProposalBasedSampler.proposal.setter 
    def proposal(self, value):
        fail_msg = "Proposal should be either None, cuqi.distribution.Distribution conditioned only on 'location' and 'scale', lambda function, or cuqi.distribution.Normal conditioned only on 'mean' and 'std'"

        if value is None:
            self._proposal = cuqi.distribution.Normal(mean = lambda location:location,std = lambda scale:scale )

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
        target_eval[0] = self.target.logpdf(self.x0)
        acc[:, 0] = np.ones(self.dim)

        # run MCMC
        for s in range(Ns-1):
            # run component by component
            samples[:, s+1], target_eval[s+1], acc[:, s+1] = self.single_update(samples[:, s], target_eval[s])

            self._print_progress(s+2,Ns) #s+2 is the sample number, s+1 is index assuming x0 is the first sample

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
        target_eval[0] = self.target.logpdf(self.x0)
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
            target_eval_star = self.target.logpdf(x_star)

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

#===================================================================
#===================================================================
#===================================================================
class MetropolisHastings(ProposalBasedSampler):
    """Metropolis Hastings sampler.

    Allows sampling of a target distribution by random-walk sampling of a proposal distribution along with an accept/reject step.

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

    Example
    -------
    .. code-block:: python

        # Parameters
        dim = 5 # Dimension of distribution
        mu = np.arange(dim) # Mean of Gaussian
        std = 1 # standard deviation of Gaussian

        # Logpdf function
        logpdf_func = lambda x: -1/(std**2)*np.sum((x-mu)**2)

        # Define distribution from logpdf as UserDefinedDistribution (sample and gradients also supported)
        target = cuqi.distribution.UserDefinedDistribution(dim=dim, logpdf_func=logpdf_func)

        # Set up sampler
        sampler = cuqi.sampler.MetropolisHastings(target, scale=1)

        # Sample
        samples = sampler.sample(2000)

    """
    #target,  proposal=None, scale=1, x0=None, dim=None
    #    super().__init__(target, proposal=proposal, scale=scale,  x0=x0, dim=dim)
    def __init__(self, target, proposal=None, scale=None, x0=None, dim=None):
        """ Metropolis-Hastings (MH) sampler. Default (if proposal is None) is random walk MH with proposal that is Gaussian with identity covariance"""
        super().__init__(target, proposal=proposal, scale=scale,  x0=x0, dim=dim)


    @ProposalBasedSampler.proposal.setter 
    def proposal(self, value):
        fail_msg = "Proposal should be either None, symmetric cuqi.distribution.Distribution or a lambda function."

        if value is None:
            self._proposal = cuqi.distribution.Gaussian(np.zeros(self.dim),np.ones(self.dim), np.eye(self.dim))
        elif not isinstance(value, cuqi.distribution.Distribution) and callable(value):
            raise NotImplementedError(fail_msg)
        elif isinstance(value, cuqi.distribution.Distribution) and value.is_symmetric:
            self._proposal = value
        else:
            raise ValueError(fail_msg)
        self._proposal.geometry = self.target.geometry

    def _sample(self, N, Nb):
        if self.scale is None:
            raise ValueError("Scale must be set to sample without adaptation. Consider using sample_adapt instead.")
        
        Ns = N+Nb   # number of simulations

        # allocation
        samples = np.empty((self.dim, Ns))
        target_eval = np.empty(Ns)
        acc = np.zeros(Ns, dtype=int)

        # initial state    
        samples[:, 0] = self.x0
        target_eval[0] = self.target.logpdf(self.x0)
        acc[0] = 1

        # run MCMC
        for s in range(Ns-1):
            # run component by component
            samples[:, s+1], target_eval[s+1], acc[s+1] = self.single_update(samples[:, s], target_eval[s])
            self._print_progress(s+2,Ns) #s+2 is the sample number, s+1 is index assuming x0 is the first sample

        # remove burn-in
        samples = samples[:, Nb:]
        target_eval = target_eval[Nb:]
        accave = acc[Nb:].mean()   
        print('\nAverage acceptance rate:', accave, '\n')
        #
        return samples, target_eval, accave

    def _sample_adapt(self, N, Nb):
        # Set intial scale if not set
        if self.scale is None:
            self.scale = 0.1
            
        Ns = N+Nb   # number of simulations

        # allocation
        samples = np.empty((self.dim, Ns))
        target_eval = np.empty(Ns)
        acc = np.zeros(Ns)

        # initial state    
        samples[:, 0] = self.x0
        target_eval[0] = self.target.logpdf(self.x0)
        acc[0] = 1

        # initial adaptation params 
        Na = int(0.1*N)                              # iterations to adapt
        hat_acc = np.empty(int(np.floor(Ns/Na)))     # average acceptance rate of the chains
        lambd = self.scale
        star_acc = 0.234    # target acceptance rate RW
        i, idx = 0, 0

        # run MCMC
        for s in range(Ns-1):
            # run component by component
            samples[:, s+1], target_eval[s+1], acc[s+1] = self.single_update(samples[:, s], target_eval[s])
            
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
            self._print_progress(s+2,Ns) #s+2 is the sample number, s+1 is index assuming x0 is the first sample


        # remove burn-in
        samples = samples[:, Nb:]
        target_eval = target_eval[Nb:]
        accave = acc[Nb:].mean()   
        print('\nAverage acceptance rate:', accave, 'MCMC scale:', self.scale, '\n')
        
        return samples, target_eval, accave


    def single_update(self, x_t, target_eval_t):
        # propose state
        xi = self.proposal.sample(1)   # sample from the proposal
        x_star = x_t + self.scale*xi.flatten()   # MH proposal

        # evaluate target
        target_eval_star = self.target.logpdf(x_star)

        # ratio and acceptance probability
        ratio = target_eval_star - target_eval_t  # proposal is symmetric
        alpha = min(0, ratio)

        # accept/reject
        u_theta = np.log(np.random.rand())
        if (u_theta <= alpha):
            x_next = x_star
            target_eval_next = target_eval_star
            acc = 1
        else:
            x_next = x_t
            target_eval_next = target_eval_t
            acc = 0
        
        return x_next, target_eval_next, acc


#===================================================================
#===================================================================
#===================================================================
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
        likelihood = cuqi.distribution.GaussianCov(mean=model, cov=np.ones(dim)).to_likelihood(mu)
        prior = cuqi.distribution.GaussianCov(mean=np.zeros(dim), cov=1)

        target = cuqi.distribution.Posterior(likelihood, prior)

        # Set up sampler
        sampler = cuqi.sampler.pCN(target, scale = 0.1)

        # Sample
        samples = sampler.sample(5000)
        
    """
    def __init__(self, target, scale=None, x0=None):
        super().__init__(target, x0=x0, dim=None) 
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
            self._loglikelihood = lambda x : self.likelihood.log(x)
        elif isinstance(value,tuple) and len(value)==2 and \
             (isinstance(value[0], cuqi.likelihood.Likelihood) or isinstance(value[0], cuqi.likelihood.UserDefinedLikelihood))  and \
             isinstance(value[1], cuqi.distribution.Distribution):
            self._target = value
            self._loglikelihood = lambda x : self.likelihood.log(x)
        else:
            raise ValueError(f"To initialize an object of type {self.__class__}, 'target' need to be of type 'cuqi.distribution.Posterior'.")
        
        #TODO:
        #if not isinstance(self.prior,(cuqi.distribution.Gaussian,cuqi.distribution.GaussianCov, cuqi.distribution.GaussianPrec, cuqi.distribution.GaussianSqrtPrec, cuqi.distribution.Normal)):
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