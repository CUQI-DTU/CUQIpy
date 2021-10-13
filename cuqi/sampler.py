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
import warnings

# another implementation is in https://github.com/mfouesneau/NUTS
class NUTS(object):

    def __init__(self, likelihood, prior, data, x0, maxdepth=20):
        self.likelihood = likelihood
        self.prior = prior
        self.data = data
        self.x0 = x0
        self.maxdepth = maxdepth

    def potential(self,x):
        """Potential of likelihood+prior. Also returns the gradient"""
        logpdf = -self.likelihood(x=x).logpdf(self.data) - self.prior.logpdf(x)
        grad = -self.likelihood.gradient(self.data,x=x) - self.prior.gradient(x)
        return logpdf, grad

    def sample(self, N, Nb):
        # Save dimension of prior
        d = self.prior.dim

        # Allocation
        Ns = Nb+N               # total number of chains
        theta = np.empty((d, Ns))
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
                logalpha = min(0, np.log(n_p) - np.log(n))
                if (s_p == 1) and (np.log(np.random.rand()) <= logalpha) and (np.isnan(pot_p) == False):
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
                
            # msg
            if (np.mod(k, 25) == 0):
                print("\nSample {:d}/{:d}".format(k, Ns))
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
            return np.random.normal(size=self.prior.dim)

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
            alpha_p = min(1, np.exp(H_p - H))    # logalpha_p = min(0, H_p - H)

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
                logalpha2 = np.log(n_pp) - np.log(n_p+n_pp) # n_pp/max(1, n_p+n_pp)
                if (np.log(np.random.rand()) <= logalpha2):
                    theta_p = np.copy(theta_pp)
                    pot_p = np.copy(pot_pp)
                    grad_pot_p = np.copy(grad_pot_pp)

                # update number of particles and stopping criterion
                alpha_p += alpha_pp
                n_alpha_p += n_alpha_pp
                n_p += n_pp   
                s_p = s_pp*(((theta_plus-theta_minus)@r_minus)>=0)*(((theta_plus-theta_minus)@r_plus)>=0)

            return theta_minus, r_minus, grad_pot_minus, theta_plus, r_plus, grad_pot_plus, theta_p, pot_p, grad_pot_p, n_p, s_p, alpha_p, n_alpha_p






#===================================================================
#===================================================================
#===================================================================
class Sampler(ABC):

    def sample(self,N,Nb):
        # Get samples from the distribution sample method
        s,loglike_eval, accave = self._sample(N,Nb)
        return self._create_Sample_object(s,N),loglike_eval, accave

    def sample_adapt(self,N,Nb):
        # Get samples from the distribution sample method
        s,loglike_eval, accave = self._sample_adapt(N,Nb)
        return self._create_Sample_object(s,N),loglike_eval, accave

    def _create_Sample_object(self,s,N):
        #Store samples in cuqi samples object if more than 1 sample
        if N==1:
            if len(s) == 1 and isinstance(s,np.ndarray): #Extract single value from numpy array
                s = s.ravel()[0]
            else:
                s = s.flatten()
        else:
            s = Samples(s)

        return s

    @abstractmethod
    def _sample(self,N,Nb):
        pass


class Linear_RTO(object):
    
    def __init__(self, likelihood, prior, model, data, x0, maxit=10, tol=1e-6, shift=0):
        
        # independent Posterior samples. Linear model and Gaussian prior-likelihood
        if not isinstance(model, cuqi.model.LinearModel):
            raise TypeError("Model needs to be linear")

        if not isinstance(likelihood, cuqi.distribution.GaussianCov):
            raise TypeError("Likelihood needs to be GaussianCov")

        if not isinstance(prior, cuqi.distribution.GaussianCov): #TODO add support for other Gaussians
            raise TypeError("Prior needs to be GaussianCov")
    
        # Extract lambda, delta, L
        #self.lambd = 1/(likelihood.std**2)
        #self.delta = prior.prec
        #self.L = prior.L
        #self.A = model.get_matrix()
        #self.b = data        
        self.x0 = x0
        self.maxit = maxit
        self.tol = tol        
        self.shift = 0
                
        L1 = likelihood.sqrtprec
        L2 = prior.sqrtprec

        # pre-computations
        self.m = len(data)
        self.n = len(x0)
        self.b_tild = np.hstack([L1@data, L2@prior.mean]) 

        self.model = model

        if not callable(model):
            self.M = sp.sparse.vstack([L1@model, L2])
        else:
            # in this case, model is a function doing forward and backward operations
            def M(x, flag):
                if flag == 1:
                    out1 = L1 @ model.forward(x)
                    out2 = L2 @ x
                    out  = np.hstack([out1, out2])
                elif flag == 2:
                    idx = int(len(x) - self.n)
                    out1 = model.adjoint(L1.T@x[:idx])
                    out2 = L2.T @ x[idx:]
                    out  = out1 + out2                
                return out   
            self.M = M       

    def sample(self, N, Nb):   
        Ns = N+Nb   # number of simulations        
        samples = np.empty((self.n, Ns))
                     
        # initial state   
        samples[:, 0] = self.x0
        for s in range(Ns-1):
            y = self.b_tild + np.random.randn(self.m+self.n)
            sim = CGLS(self.M, y, samples[:, s], self.maxit, self.tol, self.shift)            
            samples[:, s+1], _ = sim.solve()
            if (s % 1) == 0 or s == Ns-1:
                print('Sample', s+2, '/', Ns)
        
        # remove burn-in
        samples = samples[:, Nb:]
        
        return cuqi.samples.Samples(samples,geometry=self.model.domain_geometry)



#===================================================================
#===================================================================
#===================================================================
class CWMH(object):

    def __init__(self, pi_target, proposal, scale, init_x):
        self.target = pi_target
        self.proposal = proposal
        self.scale = scale
        self.x0 = init_x
        self.n = len(init_x)

    def sample(self, N, Nb):
        Ns = N+Nb   # number of simulations

        # allocation
        samples = np.empty((self.n, Ns))
        target_eval = np.empty(Ns)
        acc = np.zeros((self.n, Ns), dtype=int)

        # initial state    
        samples[:, 0] = self.x0
        target_eval[0] = self.target(self.x0)
        acc[:, 0] = np.ones(self.n)

        # run MCMC
        for s in range(Ns-1):
            # run component by component
            samples[:, s+1], target_eval[s+1], acc[:, s+1] = self.single_update(samples[:, s], target_eval[s])
            if (s % 5e2) == 0:
                print('Sample', s, '/', Ns)

        # remove burn-in
        samples = samples[:, Nb:]
        target_eval = target_eval[Nb:]
        acccomp = acc[:, Nb:].mean(axis=1)   
        print('\nAverage acceptance rate all components:', acccomp.mean(), '\n')
        
        return samples, target_eval, acccomp

    def sample_adapt(self, N, Nb):
        # this follows the vanishing adaptation Algorithm 4 in:
        # Andrieu and Thoms (2008) - A tutorial on adaptive MCMC
        Ns = N+Nb   # number of simulations

        # allocation
        samples = np.empty((self.n, Ns))
        target_eval = np.empty(Ns)
        acc = np.zeros((self.n, Ns), dtype=int)

        # initial state
        samples[:, 0] = self.x0
        target_eval[0] = self.target(self.x0)
        acc[:, 0] = np.ones(self.n)

        # initial adaptation params 
        Na = int(0.1*N)                                        # iterations to adapt
        hat_acc = np.empty((self.n, int(np.floor(Ns/Na))))     # average acceptance rate of the chains
        lambd = np.empty((self.n, int(np.floor(Ns/Na)+1)))     # scaling parameter \in (0,1)
        lambd[:, 0] = self.scale
        star_acc = 0.21/self.n + 0.23    # target acceptance rate RW
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
                self.scale = np.minimum(lambd[:, i+1], np.ones(self.n))

                # update counters
                i += 1
                idx += Na

            # display iterations 
            if ((s % 5e2) == 0 and s < Nb) or s ==  Nb:
                print('Burn-in', s, '/', Nb)
            elif (s % 5e2) == 0:
                print('Sample', s-Nb, '/', N)

        # remove burn-in
        samples = samples[:, Nb:]
        target_eval = target_eval[Nb:]
        acccomp = acc[:, Nb:].mean(axis=1)
        print('\nAverage acceptance rate all components:', acccomp.mean(), '\n')
        
        return samples, target_eval, acccomp

    def single_update(self, x_t, target_eval_t):
        x_i_star = self.proposal(x_t, self.scale)
        x_star = x_t.copy()
        acc = np.zeros(self.n)

        for j in range(self.n):
            # propose state
            x_star[j] = x_i_star[j]

            # evaluate target
            target_eval_star = self.target(x_star)

            # ratio and acceptance probability
            ratio = np.exp(target_eval_star - target_eval_t)  # proposal is symmetric
            alpha = min(1, ratio)

            # accept/reject
            u_theta = np.random.rand()
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
class MetropolisHastings(Sampler):

    def __init__(self, target, proposal=None, scale=1, x0=None):
        """ Metropolis-Hastings (MH) sampler. Default (if proposal is None) is random walk MH with proposal that is Gaussian with identity covariance"""
        self.dim = target.dim

        if proposal is None:
            proposal = cuqi.distribution.Gaussian(np.zeros(self.dim),np.ones(self.dim), np.eye(self.dim))
        elif not proposal.is_symmetric:
            raise ValueError("Proposal needs to be a symmetric distribution")

        if x0 is None:
            x0 = np.ones(self.dim)

        self.proposal =proposal
        self.target = target
        self.scale = scale
        self.x0 = x0


    def _sample(self, N, Nb):
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
            if (s % 5e2) == 0:
                print('Sample', s, '/', Ns)

        # remove burn-in
        samples = samples[:, Nb:]
        target_eval = target_eval[Nb:]
        accave = acc[Nb:].mean()   
        print('\nAverage acceptance rate:', accave, '\n')
        #
        return samples, target_eval, accave

    def _sample_adapt(self, N, Nb):
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
            if (s % 5e2) == 0:
                print('Sample', s, '/', Ns)

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
        ratio = np.exp(target_eval_star - target_eval_t)  # proposal is symmetric
        alpha = min(1, ratio)

        # accept/reject
        u_theta = np.random.rand()
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
    
    def __init__(self, logprior, loglike, scale, init_x):
        self.prior = logprior   # this is used in the proposal and must be a Gaussian
        self.loglike = loglike
        self.scale = scale
        self.x0 = init_x
        self.n = len(init_x)

    def _sample(self, N, Nb):
        Ns = N+Nb   # number of simulations

        # allocation
        samples = np.empty((self.n, Ns))
        loglike_eval = np.empty(Ns)
        acc = np.zeros(Ns, dtype=int)

        # initial state    
        samples[:, 0] = self.x0
        loglike_eval[0] = self.loglike(self.x0)
        acc[0] = 1

        # run MCMC
        for s in range(Ns-1):
            # run component by component
            samples[:, s+1], loglike_eval[s+1], acc[s+1] = self.single_update(samples[:, s], loglike_eval[s])
            if (s % 5e2) == 0:
                print('Sample', s, '/', Ns)

        # remove burn-in
        samples = samples[:, Nb:]
        loglike_eval = loglike_eval[Nb:]
        accave = acc[Nb:].mean()   
        print('\nAverage acceptance rate:', accave, '\n')
        #
        return samples, loglike_eval, accave

    def _sample_adapt(self, N, Nb):
        Ns = N+Nb   # number of simulations

        # allocation
        samples = np.empty((self.n, Ns))
        loglike_eval = np.empty(Ns)
        acc = np.zeros(Ns)

        # initial state    
        samples[:, 0] = self.x0
        loglike_eval[0] = self.loglike(self.x0)
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
            if (s % 5e2) == 0:
                print('Sample', s, '/', Ns)

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
        loglike_eval_star = self.loglike(x_star)

        # ratio and acceptance probability
        ratio = np.exp(loglike_eval_star - loglike_eval_t)  # proposal is symmetric
        alpha = min(1, ratio)

        # accept/reject
        u_theta = np.random.rand()
        if (u_theta <= alpha):
            x_next = x_star
            loglike_eval_next = loglike_eval_star
            acc = 1
        else:
            x_next = x_t
            loglike_eval_next = loglike_eval_t
            acc = 0
        
        return x_next, loglike_eval_next, acc