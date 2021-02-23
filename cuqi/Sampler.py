import scipy as sp
import scipy.stats as sps
import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
eps = np.finfo(float).eps

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
            if (s % 5e2) == 0:
                print('Sample', s, '/', Ns)

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
class RWMH(object):

    def __init__(self, logprior, loglike, scale, init_x):
        self.prior = logprior   # this works as proposal and must be a Gaussian
        self.loglike = loglike
        self.scale = scale
        self.x0 = init_x
        self.n = len(init_x)

    def sample(self, N, Nb):
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

    def sample_adapt(self, N, Nb):
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
        star_acc = 0.234    # target acceptance rate RW
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
        xi = self.prior.sample(1)   # sample from the Gaussian prior
        x_star = x_t + self.scale*xi   # pCN proposal

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


#===================================================================
#===================================================================
#===================================================================
class pCN(object):    
    
    def __init__(self, logprior, loglike, scale, init_x):
        self.prior = logprior   # this is used in the proposal and must be a Gaussian
        self.loglike = loglike
        self.scale = scale
        self.x0 = init_x
        self.n = len(init_x)

    def sample(self, N, Nb):
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

    def sample_adapt(self, N, Nb):
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
        xi = self.prior.sample(1)   # sample from the prior
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