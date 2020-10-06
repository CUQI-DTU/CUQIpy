# ========================================================================
# Created by:
# Felipe Uribe @ DTU compute
# ========================================================================
# Version 2020-06
# ========================================================================
import scipy as sp
import scipy.stats as sps
import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
eps = np.finfo(float).eps



#=========================================================================
#=========================================================================
#=========================================================================
def CWMH(Ns, pi_target, proposal, x0, scale):
    n = len(x0)
    
    # allocation
    samples = np.empty((n, Ns))
    target_eval = np.empty(Ns)
    acc = np.zeros((n, Ns))

    # initial state    
    samples[:,0] = x0
    target_eval[0] = pi_target(x0)
    acc[:,0] = np.ones(n)

    # run MCMC
    for s in range(Ns-1):
        # run component by component
        samples[:,s+1], target_eval[s+1], acc[:,s+1] = CW_MH_single(n, samples[:,s], target_eval[s], pi_target, proposal, scale)
        
        if (s % 5e2) == 0:
            print('Sample', s, '/', Ns)
                  
    return samples, target_eval, acc

#=========================================================================
def CW_MH_single(n, x_t, target_eval_t, pi_target, proposal, scale):
    #
    x_i_star = proposal(1, x_t, scale)
    x_star = np.copy(x_t)
    acc = np.zeros(n)
    #
    for j in range(n):
        # propose state
        x_star[j] = x_i_star[0,j]
        
        # evaluate target
        target_eval_star = pi_target(x_star)
        
        # ratio and acceptance probability
        ratio = np.exp(target_eval_star - target_eval_t)  # proposal is symmetric
        alpha = min(1, ratio)

        # accept/reject
        u_theta = np.random.rand()
        if (u_theta <= alpha):
            x_t[j] = x_i_star[0,j]
            target_eval_t = target_eval_star
            acc[j] = 1
        else:
            pass
            # x_t[j]       = x_t[j]
            # target_eval_t = target_eval_t
        x_star = np.copy(x_t)
        
    return x_t, target_eval_t, acc



#=========================================================================
#=========================================================================
#=========================================================================
def sample_RW(n, Nc, Nb, beta, u0, pi_target):
    # allocation
    NN             = Nb + Nc            # total number of chains
    theta          = np.empty((n, NN))
    pi_target_eval = np.empty(NN)
    acc            = np.zeros(NN)

    # initial state
    theta[:,0]        = u0
    pi_target_eval[0] = pi_target(theta[:,0])
    acc[0]            = 1
    
    for t in range(NN-1):
        # current state
        t_t       = theta[:,t]
        pi_eval_t = pi_target_eval[t]
    
        # propose state
        t_star = t_t + beta*np.random.randn(n)
    
        if (t_star >= 0):    
            # evaluate target
            pi_eval_star = pi_target(t_star)
    
            # ratio and acceptance probability
            ratio = np.exp(pi_eval_star - pi_eval_t)  # proposal is symmetric
            alpha = min(1, ratio)
        else:
            alpha = 0
    
        # accept/reject
        u_theta = np.random.rand()
        if (u_theta <= alpha):
            theta[:,t+1]        = t_star
            pi_target_eval[t+1] = pi_eval_star
            acc[t+1]            = 1
        else:
            theta[:,t+1]        = t_t
            pi_target_eval[t+1] = pi_eval_t
            
    return theta