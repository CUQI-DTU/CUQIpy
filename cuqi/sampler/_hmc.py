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
    def __init__(self, target, x0=None, traject_length=15, adapt_step_size=True, opt_acc_rate=0.65, **kwargs):
        super().__init__(target, x0=x0, **kwargs)
        self.traject_length = traject_length
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

    def _update_run_diagnostic_attributes(self, k, eps, eps_bar):
        """A method to update attributes to store run diagnostic."""
        # Store the current iteration number k
        self.iteration_list.append(k)
        # Store the number of tree nodes created in iteration k
        self.epsilon_list.append(eps)
        # Store the step size suggestion during adaptation in iteration k
        self.epsilon_bar_list.append(eps_bar)

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

        # Step size variables
        epsilon, epsilon_bar = None, None

        # parameters dual averaging
        delta = self.opt_acc_rate # https://mc-stan.org/docs/2_18/reference-manual/hmc-algorithm-parameters.html
        if (self.adapt_step_size == True):
            epsilon = self._FindGoodEpsilon(theta[:, 0], joint_eval[0], grad)
            mu = np.log(10*epsilon)
            gamma, t_0, kappa = 0.05, 10, 0.75 # kappa in (0.5, 1]
            epsilon_bar, H_bar = 1, 0
            step_sizes[0] = epsilon
        elif (self.adapt_step_size == False):
            epsilon = self._FindGoodEpsilon(theta[:, 0], joint_eval[0], grad)
        else:
            epsilon = self.adapt_step_size # if scalar then user specifies the step size

        # set the trajectory length as it has to be accessed many times
        if (self.adapt_traject_length == True):
            # the best simulation length 'lambd' was 0.14-18; reported in some studies, 
            # but this is a tricky parameter and finding a good 'lambd' for HMC requires
            # some number of preliminary runs (that is why NUTS was created)
            lambd = 10 
            L = np.max(1, np.round(lambd/epsilon))
        else:
            L = self.adapt_traject_length # if scalar then user specifies the step size
        traject_lengths[0] = L

        # run HMC
        for k in range(Ns-1):
            q_k = theta[:, k]                # initial position (parameters)
            p_k = self._Kfun(1, 'sample')    # initial momentum vector

            # LEAPFROG: alternate full steps for position and momentum
            q_star, p_star = self._Leapfrog(q_k, p_k, grad, epsilon, L)

            # evaluate neg-potential and neg-kinetic energies at start and end of trajectory
            U_k = joint_eval[k]
            U_star, grad_star = self._hmc_target(q_star)
            K_star, K_k = self._Kfun(p_star, 'eval'), self._Kfun(p_k, 'eval')

            # accept/reject
            log_alpha = min( 0, (U_star+K_star)-(U_k+K_k) )
            log_u = np.log(np.random.rand())
            if (log_u <= log_alpha):
                theta[:, k+1] = q_star
                joint_eval[k+1] = U_star
                acc[k+1] = 1
                grad = grad_star
            else:
                theta[:, k+1] = q_k
                joint_eval[k+1] = U_k

            # update run diagnostic attributes
            self._update_run_diagnostic_attributes(k, epsilon, epsilon_bar)

            # adapt epsilon during burn-in using dual averaging
            if (k <= Nb) and (self.adapt_step_size == True):
                eta1 = 1/(k + t_0)
                H_bar = (1-eta1)*H_bar + eta1*(delta - np.exp(log_alpha))
                epsilon = np.exp(mu - (np.sqrt(k)/gamma)*H_bar)
                eta = k**(-kappa)
                epsilon_bar = np.exp(eta*np.log(epsilon) + (1-eta)*np.log(epsilon_bar))
            elif (k == Nb+1) and (self.adapt_step_size == True):
                # after warm-up we jitter the epsilons to avoid pathological behavior, see Neal's reference
                epsilon = np.random.uniform(0.9*epsilon_bar, 1.1*epsilon_bar)
            step_sizes[k] = epsilon

            # adapt path length
            if (k <= Nb) and (self.adapt_traject_length == True):
                L = np.max(1, np.round(lambd/epsilon))
            traject_lengths[k] = L

            # msg
            self._print_progress(k+1, Ns) #k+1 is the sample number, k is index assuming x0 is the first sample
            self._call_callback(theta[:, k], k)

            if np.isnan(joint_eval[k]):
                raise NameError('NaN potential func')

        # apply burn-in
        if not self._return_burnin: 
            theta = theta[:, Nb:]
            joint_eval = joint_eval[Nb:]
        acc_rate = np.mean(acc)

        return theta, joint_eval, step_sizes, traject_lengths, acc_rate

    #=========================================================================
    # auxiliary standard Gaussian PDF: neg-kinetic energy function
    # d_log_2pi = d*np.log(2*np.pi)
    def _Kfun(self, r, flag):
        if flag == 'eval': # evaluate
            return -0.5*(r.T @ r) #- d_log_2pi 
        if flag == 'sample': # sample
            return np.random.standard_normal(size=self.dim)

    #=========================================================================
    def _FindGoodEpsilon(self, theta, joint, grad, epsilon=1):
        r = self._Kfun(1, 'sample')    # resample a momentum
        Ham = joint - self._Kfun(r, 'eval')     # initial Hamiltonian
        _, r_prime, joint_prime, grad_prime = self._Leapfrog(theta, r, grad, epsilon)

        # trick to make sure the step is not huge, leading to infinite values of the likelihood
        k = 1
        while np.isinf(joint_prime) or np.isinf(grad_prime).any():
            k *= 0.5
            _, r_prime, joint_prime, grad_prime = self._Leapfrog(theta, r, grad, epsilon*k)
        epsilon = 0.5*k*epsilon

        # doubles/halves the value of epsilon until the accprob of the Langevin proposal crosses 0.5
        Ham_prime = joint_prime - self._Kfun(r_prime, 'eval')
        log_ratio = Ham_prime - Ham
        a = 1 if log_ratio > np.log(0.5) else -1
        while (a*log_ratio > -a*np.log(2)):
            epsilon = (2**a)*epsilon
            _, r_prime, joint_prime, _ = self._Leapfrog(theta, r, grad, epsilon)
            Ham_prime = joint_prime - self._Kfun(r_prime, 'eval')
            log_ratio = Ham_prime - Ham
        return epsilon

    #=========================================================================
    def _Leapfrog(self, q0, p0, grad_old, epsilon, L):
        # symplectic integrator: trajectories preserve phase space volumen
        # faster: do not store trajectory
        q, p = np.copy(q0), np.copy(p0)
        p += 0.5*epsilon*grad_old  # initial half step for momentum
        for n in range(L):
            q += epsilon*p   # full step for the position
            grad_q = self._hmc_target(q)[1]
            if (n != L-1):
                p += epsilon*grad_q   # full step for the momentum, skip last one
        p += (epsilon/2)*grad_q       # final half step for momentum 

        return q, p
