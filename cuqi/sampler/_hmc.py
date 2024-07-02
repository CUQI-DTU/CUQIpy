import numpy as np
from cuqi.sampler import Sampler


class HMC(Sampler):
    """Hamiltonian Monte Carlo (Duane et al., 1987).

    Samples a distribution given its logpdf and gradient using a Hamiltonian Monte Carlo (HMC) algorithm with automatic parameter tuning.

    For more details see:
    Neal, R. M. (2011) - MCMC Using Hamiltonian Dynamics. Handbook of Markov chain Monte Carlo. Ed. by S. Brooks et al. Chapman & Hall/CRC, 2011. Chap. 5, pp. 113-162.
    Duane, S., Kennedy, A. D., Pendleton, B. J., and Roweth, D. 1987. Hybrid Monte Carlo. Physics Letters B, 195:216-222.

    Parameters
    ----------

    target : `cuqi.distribution.Distribution`
        The target distribution to sample. Must have logpdf and gradient method. Custom logpdfs and gradients are supported by using a :class:`cuqi.distribution.UserDefinedDistribution`.

    x0 : ndarray
        Initial parameters. *Optional*

    adapt_traject_length : int
        This is the trajectory length or steps in the numerical integrator (leapfrog).
        If True, the step size is adapted.
        If set to a scalar, the step size will be given by user and not adapted.

    adapt_step_size : Bool or float
        Whether to adapt the step size.
        If True, the step size is adapted automatically.
        If False, the step size is fixed to the initially estimated value.
        If set to a scalar, the step size will be given by user and not adapted.

    opt_acc_rate : float
        The optimal acceptance rate to reach if using adaptive step size.
        Suggested value is 0.65 (default).

    callback : callable, *Optional*
        If set this function will be called after every sample.
        The signature of the callback function is `callback(sample, sample_index)`,
        where `sample` is the current sample and `sample_index` is the index of the sample.
        An example is shown in demos/demo31_callback.py.

    Example
    -------
    .. code-block:: python

        # Import cuqi
        import cuqi

        # Define a target distribution
        tp = cuqi.testproblem.WangCubic()
        target = tp.posterior

        # Set up sampler
        sampler = cuqi.sampler.NUTS(target)

        # Sample
        samples = sampler.sample(10000, 5000)

        # Plot samples
        samples.plot_pair()

    After running the NUTS sampler, run diagnostics can be accessed via the 
    following attributes:

    .. code-block:: python

        # Number of tree nodes created each NUTS iteration
        sampler.num_tree_node_list

        # Step size used in each NUTS iteration
        sampler.epsilon_list

        # Suggested step size during adaptation (the value of this step size is
        # only used after adaptation). The suggested step size is None if 
        # adaptation is not requested.
        sampler.epsilon_bar_list

        # Additionally, iterations' number can be accessed via
        sampler.iteration_list

    """
    def __init__(self, target, x0=None, adapt_traject_length=10, adapt_step_size=True, opt_acc_rate=0.8, **kwargs):
        super().__init__(target, x0=x0, **kwargs)
        self.adapt_traject_length = adapt_traject_length
        self.adapt_step_size = adapt_step_size
        self.opt_acc_rate = opt_acc_rate
        # if this flag is True, the samples and the burn-in will be returned
        # otherwise, the burn-in will be truncated
        self._return_burnin = False

        # run diagnostic
        # Create lists to store HMC run diagnostics
        self._create_run_diagnostic_attributes()

    def _create_run_diagnostic_attributes(self):
        """A method to create attributes to store run diagnostic."""
        self._reset_run_diagnostic_attributes()

    def _reset_run_diagnostic_attributes(self):
        """A method to reset attributes to store run diagnostic."""
        # iterations
        self.iteration_list = []
        # List of step size used in each iteration 
        self.epsilon_list = []
        # List of burn-in step size suggestion during adaptation 
        # only used when adaptation is done
        # remains fixed after adaptation (after burn-in)
        self.epsilon_bar_list = []
        self.traject_length_list = []

    def _update_run_diagnostic_attributes(self, k, eps, eps_bar, L):
        """A method to update attributes to store run diagnostic."""
        # Store the current iteration number k
        self.iteration_list.append(k)
        # Store the number of tree nodes created in iteration k
        self.epsilon_list.append(eps)
        # Store the step size suggestion during adaptation in iteration k
        self.epsilon_bar_list.append(eps_bar)
        # Store the trajectory length during adaptation in iteration k
        self.traject_length_list.append(L)

    def _hmc_target(self, x): # returns logposterior tuple evaluation-gradient
        return self.target.logd(x), self.target.gradient(x)

    def _sample_adapt(self, N, Nb):
        return self._sample(N, Nb)

    def _sample(self, N, Nb):
        # Reset run diagnostic attributes
        self._reset_run_diagnostic_attributes()
        
        if self.adapt_step_size is True and Nb == 0:
            raise ValueError("Adaptive step size is True but number of burn-in steps is 0. Please set Nb > 0.")

        # Allocation
        Ns = Nb+N     # total number of chains
        theta = np.empty((self.dim, Ns))
        joint_eval = np.empty(Ns)
        step_sizes = np.empty(Ns)
        traject_lengths = np.empty(Ns, dtype=int)
        acc = np.zeros(Ns, dtype=int)

        # Initial state
        theta[:, 0] = self.x0
        joint_eval[0], grad = self._hmc_target(self.x0)

        # set the trajectory length as it has to be accessed many times
        if (self.adapt_traject_length == True):
            epsilon = 1e-2  # arbitrary initialization
            # the best simulation length 'lambd' was 0.14-18; reported in some studies, 
            # but this is a tricky parameter and finding a good 'lambd' for HMC requires
            # some number of preliminary runs (that is why NUTS was created)
            lambd = 10 
            L = np.max(1, np.round(lambd/epsilon))
        else:
            L = self.adapt_traject_length # if scalar then user specifies the step size
        traject_lengths[0] = L

        # parameters dual averaging        
        epsilon, epsilon_bar = None, None
        if (self.adapt_step_size == True):
            epsilon = self._FindGoodEpsilon(theta[:, 0], joint_eval[0], grad, L)
            mu = np.log(10*epsilon)
            log_epsilon_bar, H_bar = 0, 0
            gamma, t_0, kappa = 0.05, 10, 0.75 # kappa in (0.5, 1]
            delta = self.opt_acc_rate # https://mc-stan.org/docs/2_18/reference-manual/hmc-algorithm-parameters.html
        elif (self.adapt_step_size == False):
            epsilon = self._FindGoodEpsilon(theta[:, 0], joint_eval[0], grad, L)
        else:
            epsilon = self.adapt_step_size # if scalar then user specifies the step size
        step_sizes[0] = epsilon

        # run HMC
        for k in range(1, Ns):
            q_k = theta[:, k-1]                # initial position (parameters)
            r_k = self._neg_kinetic_func(1, 'sample')    # initial momentum vector

            # LEAPFROG: alternate full steps for position and momentum
            q_star, r_star, joint_star, grad_star = self._Leapfrog(q_k, r_k, grad, epsilon, L)

            # evaluate neg-potential and neg-kinetic energies at start and end of trajectory
            joint_k = joint_eval[k-1]
            nHam_k = self._neg_Hamiltonian(joint_k, r_k) # Hamiltonian
            nHam_star = self._neg_Hamiltonian(joint_star, r_star) # Hamiltonian

            # accept/reject
            log_alpha = -1000 if np.isneginf(nHam_star) else (0 if np.isinf(nHam_star) else min(0, nHam_star - nHam_k))
            log_u = np.log(np.random.rand())
            if (log_u <= log_alpha) and not np.isnan(nHam_star):
                theta[:, k] = q_star
                joint_eval[k] = joint_star
                acc[k] = 1
                grad = grad_star
            else:
                theta[:, k] = q_k
                joint_eval[k] = joint_k

            # update run diagnostic attributes
            self._update_run_diagnostic_attributes(k, epsilon, epsilon_bar, L)

            # adapt epsilon during burn-in using dual averaging
            if (k <= Nb) and (self.adapt_step_size == True):
                eta1, eta2 = 1/(k+t_0), k**(-kappa)
                #
                H_bar = (1-eta1)*H_bar + eta1*(delta - np.exp(log_alpha))
                log_epsilon = mu - (np.sqrt(k)/gamma)*H_bar                
                log_epsilon_bar = eta2*log_epsilon + (1-eta2)*log_epsilon_bar
                #
                epsilon, epsilon_bar = np.exp(log_epsilon), np.exp(log_epsilon_bar)
            elif (k > Nb) and (self.adapt_step_size == True):
                # after warm-up we jitter the epsilons to avoid pathological behavior, see Neal's reference
                epsilon = np.random.uniform(0.9*epsilon_bar, 1.1*epsilon_bar)
            step_sizes[k] = epsilon

            # adapt path length
            if (k <= Nb) and (self.adapt_traject_length == True):
                L = np.max(1, np.round(lambd/epsilon))
            traject_lengths[k] = L

            # msg
            self._print_progress(k, Ns) #k+1 is the sample number, k is index assuming x0 is the first sample
            self._call_callback(theta[:, k], k)

        # apply burn-in
        if not self._return_burnin: 
            theta = theta[:, Nb:]
            joint_eval = joint_eval[Nb:]
        acc_rate = np.mean(acc)
        print('\tAcceptance rate:', acc_rate, '\n')

        return theta, joint_eval, step_sizes

    #=========================================================================
    # kinetic energy function (negative log PDF): assumed to be standard Gaussian PDF
    # d_log_2pi = d*np.log(2*np.pi)
    # here we implement the negative kinetic fun
    def _neg_kinetic_func(self, r, flag):
        if flag == 'eval': # evaluate
            return -0.5*(r.T @ r) #- d_log_2pi 
        if flag == 'sample': # sample
            return np.random.standard_normal(size=self.dim)

    #=========================================================================
    # Hamiltonian function (negative log)
    def _neg_Hamiltonian(self, nU, r):
        # here nU is the log-posterior (so it is the negative potential) 
        # and we work with negative kinetic fun, but the Hamiltonian is:
        # H(q,r) = U(q) + K(r)
        # U(q): potential energy: negative log-posterior
        # K(r): kinetic energy: negative log-assumed-density
        nK = self._neg_kinetic_func(r, 'eval')

        return nU + nK

    #=========================================================================
    def _FindGoodEpsilon(self, theta, joint, grad, L, epsilon=1):
        r = self._neg_kinetic_func(1, 'sample')         # resample a momentum
        nHam = self._neg_Hamiltonian(joint, r)   # initial Hamiltonian
        _, r_prime, joint_prime, grad_prime = self._Leapfrog(theta, r, grad, epsilon, L)

        # trick to make sure the step is not huge, leading to infinite values of the likelihood
        k = 1
        while np.isnan(joint_prime) or np.isnan(grad_prime).any() or np.isinf(joint_prime) or np.isinf(grad_prime).any():
            k *= 0.5
            _, r_prime, joint_prime, grad_prime = self._Leapfrog(theta, r, grad, epsilon*k, L)
        epsilon = 0.5*k*epsilon

        # doubles/halves the value of epsilon until the accprob of the Langevin proposal crosses 0.5
        nHam_prime = self._neg_Hamiltonian(joint_prime, r_prime)
        log_ratio = nHam_prime - nHam
        a = 1 if log_ratio > np.log(0.5) else -1
        while (a*log_ratio > -a*np.log(2)):
            epsilon = (2**a)*epsilon
            _, r_prime, joint_prime, _ = self._Leapfrog(theta, r, grad, epsilon, L)
            nHam_prime = self._neg_Hamiltonian(joint_prime, r_prime)
            log_ratio = nHam_prime - nHam

        return epsilon

    #=========================================================================
    def _Leapfrog(self, q0, r0, grad, epsilon, L):
        # symplectic integrator: trajectories preserve phase space volumen
        q, r = np.copy(q0), np.copy(r0)
        for n in range(L):
            q, r, joint, grad = self._Leapfrog_single(q, r, grad, epsilon)

        return q, -r, joint, grad  # negate momentum to make proposal symmetric

    #=========================================================================
    def _Leapfrog_single(self, theta_old, r_old, grad_old, epsilon):
        # single-step update
        r_new = r_old + 0.5*epsilon*grad_old     # half-step
        theta_new = theta_old + epsilon*r_new     # full-step
        joint_new, grad_new = self._hmc_target(theta_new)     # new gradient
        r_new += 0.5*epsilon*grad_new     # half-step

        return theta_new, r_new, joint_new, grad_new
