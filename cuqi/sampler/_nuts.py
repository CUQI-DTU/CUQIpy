import numpy as np
from cuqi.sampler import Sampler


# another implementation is in https://github.com/mfouesneau/NUTS
class NUTS(Sampler):
    """No-U-Turn Sampler (Hoffman and Gelman, 2014).

    Samples a distribution given its logpdf and gradient using a Hamiltonian Monte Carlo (HMC) algorithm with automatic parameter tuning.

    For more details see: 
    Hoffman, M. D., & Gelman, A. (2014). The no-U-turn sampler: Adaptively setting path lengths in Hamiltonian Monte Carlo. Journal of Machine Learning Research, 15, 1593-1623.

    Parameters
    ----------

    target : `cuqi.distribution.Distribution`
        The target distribution to sample. Must have logpdf and gradient method. Custom logpdfs and gradients are supported by using a :class:`cuqi.distribution.UserDefinedDistribution`.
    
    x0 : ndarray
        Initial parameters. *Optional*

    max_depth : int
        Maximum depth of the tree.

    adapt_step_size : Bool or float
        Whether to adapt the step size.
        If True, the step size is adapted automatically.
        If False, the step size is fixed to the initially estimated value.
        If set to a scalar, the step size will be given by user and not adapted.

    opt_acc_rate : float
        The optimal acceptance rate to reach if using adaptive step size.
        Suggested values are 0.6 (default) or 0.8 (as in stan).

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
    def __init__(self, target, x0=None, max_depth=15, adapt_step_size=True, opt_acc_rate=0.6, **kwargs):
        super().__init__(target, x0=x0, **kwargs)
        self.max_depth = max_depth
        self.adapt_step_size = adapt_step_size
        self.opt_acc_rate = opt_acc_rate
        # if this flag is True, the samples and the burn-in will be returned
        # otherwise, the burn-in will be truncated
        self._return_burnin = False

        # NUTS run diagnostic
        # number of tree nodes created each NUTS iteration
        self._num_tree_node = 0
        # Create lists to store NUTS run diagnostics
        self._create_run_diagnostic_attributes()

    def _create_run_diagnostic_attributes(self):
        """A method to create attributes to store NUTS run diagnostic."""
        self._reset_run_diagnostic_attributes()

    def _reset_run_diagnostic_attributes(self):
        """A method to reset attributes to store NUTS run diagnostic."""
        # NUTS iterations
        self.iteration_list = []
        # List to store number of tree nodes created each NUTS iteration
        self.num_tree_node_list = []
        # List of step size used in each NUTS iteration 
        self.epsilon_list = []
        # List of burn-in step size suggestion during adaptation 
        # only used when adaptation is done
        # remains fixed after adaptation (after burn-in)
        self.epsilon_bar_list = []

    def _update_run_diagnostic_attributes(self, k, n_tree, eps, eps_bar):
        """A method to update attributes to store NUTS run diagnostic."""
        # Store the current iteration number k
        self.iteration_list.append(k)
        # Store the number of tree nodes created in iteration k
        self.num_tree_node_list.append(n_tree)
        # Store the step size used in iteration k
        self.epsilon_list.append(eps)
        # Store the step size suggestion during adaptation in iteration k
        self.epsilon_bar_list.append(eps_bar)

    def _nuts_target(self, x): # returns logposterior tuple evaluation-gradient
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

        # Initial state
        theta[:, 0] = self.x0
        joint_eval[0], grad = self._nuts_target(self.x0)

        # Step size variables
        epsilon, epsilon_bar = None, None

        # parameters dual averaging
        if (self.adapt_step_size == True):
            epsilon = self._FindGoodEpsilon(theta[:, 0], joint_eval[0], grad)
            mu = np.log(10*epsilon)
            log_epsilon_bar, H_bar = 0, 0
            gamma, t_0, kappa = 0.05, 10, 0.75 # kappa in (0.5, 1]
            delta = self.opt_acc_rate # https://mc-stan.org/docs/2_18/reference-manual/hmc-algorithm-parameters.html
            step_sizes[0] = epsilon
        elif (self.adapt_step_size == False):
            epsilon = self._FindGoodEpsilon(theta[:, 0], joint_eval[0], grad)
        else:
            epsilon = self.adapt_step_size # if scalar then user specifies the step size

        # run NUTS
        for k in range(1, Ns):
            # reset number of tree nodes for each iteration
            self._num_tree_node = 0

            theta_k, joint_k = theta[:, k-1], joint_eval[k-1] # initial position (parameters)
            r_k = self._neg_kinetic_func(1, 'sample') # resample momentum vector
            nHam = self._neg_Hamiltonian(joint_k, r_k) # Hamiltonian

            # slice variable
            log_u = np.log(np.random.uniform(0, np.exp(nHam)))

            # initialization
            j, n, s = 0, 1, 1
            theta[:, k], joint_eval[k] = theta_k, joint_k
            theta_minus, theta_plus = np.copy(theta_k), np.copy(theta_k)
            grad_minus, grad_plus = np.copy(grad), np.copy(grad)
            r_minus, r_plus = np.copy(r_k), np.copy(r_k)

            # run NUTS
            while (s == 1) and (j <= self.max_depth):
                # sample a direction
                v = int(2*(np.random.rand() < 0.5)-1)

                # build tree: doubling procedure
                if (v == -1):
                    theta_minus, r_minus, grad_minus, _, _, _, \
                    theta_prime, joint_prime, grad_prime, n_prime, s_prime, alpha, n_alpha = \
                        self._BuildTree(theta_minus, r_minus, grad_minus, nHam, log_u, v, j, epsilon)
                else:
                    _, _, _, theta_plus, r_plus, grad_plus, \
                    theta_prime, joint_prime, grad_prime, n_prime, s_prime, alpha, n_alpha = \
                        self._BuildTree(theta_plus, r_plus, grad_plus, nHam, log_u, v, j, epsilon)

                # Metropolis step
                alpha2 = min(1, (n_prime/n)) #min(0, np.log(n_p) - np.log(n))
                if (s_prime == 1) and (np.random.rand() <= alpha2) and not np.isnan(joint_prime) and not np.isinf(joint_prime):
                    theta[:, k] = theta_prime
                    joint_eval[k] = joint_prime
                    grad = np.copy(grad_prime)

                # update number of particles, tree level, and stopping criterion
                n += n_prime
                dtheta = theta_plus - theta_minus
                s = s_prime * int((dtheta @ r_minus.T) >= 0) * int((dtheta @ r_plus.T) >= 0)
                j += 1

            # update run diagnostic attributes
            self._update_run_diagnostic_attributes(
                k, self._num_tree_node, epsilon, epsilon_bar)

            # adapt epsilon during burn-in using dual averaging
            if (k <= Nb) and (self.adapt_step_size == True):
                eta1, eta2 = 1/(k+t_0), k**(-kappa)
                #
                H_bar = (1-eta1)*H_bar + eta1*(delta - (alpha/n_alpha))
                log_epsilon = mu - (np.sqrt(k)/gamma)*H_bar                
                log_epsilon_bar = eta2*log_epsilon + (1-eta2)*log_epsilon_bar
                #
                epsilon, epsilon_bar = np.exp(log_epsilon), np.exp(log_epsilon_bar)
            elif (k == Nb+1) and (self.adapt_step_size == True):
                epsilon = epsilon_bar   # fix epsilon after burn-in
            step_sizes[k] = epsilon

            # msg
            self._print_progress(k+1, Ns) #k+1 is the sample number, k is index assuming x0 is the first sample
            self._call_callback(theta[:, k], k)

            if np.isnan(joint_eval[k]):
                raise NameError('NaN potential func')

        # apply burn-in
        if not self._return_burnin: 
            theta = theta[:, Nb:]
            joint_eval = joint_eval[Nb:]
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
    def _FindGoodEpsilon(self, theta, joint, grad, epsilon=1):
        r = self._neg_kinetic_func(1, 'sample')         # resample a momentum
        nHam = self._neg_Hamiltonian(joint, r)   # initial Hamiltonian
        _, r_prime, joint_prime, grad_prime = self._Leapfrog_single(theta, r, grad, epsilon)

        # trick to make sure the step is not huge, leading to infinite values of the likelihood
        k = 1
        while np.isnan(joint_prime) or np.isnan(grad_prime).any() or np.isinf(joint_prime) or np.isinf(grad_prime).any():
            k *= 0.5
            _, r_prime, joint_prime, grad_prime = self._Leapfrog_single(theta, r, grad, epsilon*k)
        epsilon = 0.5*k*epsilon

        # doubles/halves the value of epsilon until the accprob of the Langevin proposal crosses 0.5
        nHam_prime = self._neg_Hamiltonian(joint_prime, r_prime)
        log_ratio = nHam_prime - nHam
        a = 1 if log_ratio > np.log(0.5) else -1
        while (a*log_ratio > -a*np.log(2)):
            epsilon = (2**a)*epsilon
            _, r_prime, joint_prime, _ = self._Leapfrog_single(theta, r, grad, epsilon)
            nHam_prime = self._neg_Hamiltonian(joint_prime, r_prime)
            log_ratio = nHam_prime - nHam
        return epsilon

    #=========================================================================
    def _Leapfrog_single(self, theta_old, r_old, grad_old, epsilon):
        # symplectic integrator: trajectories preserve phase space volumen
        # single-step update
        r_new = r_old + 0.5*epsilon*grad_old     # half-step
        theta_new = theta_old + epsilon*r_new     # full-step
        joint_new, grad_new = self._nuts_target(theta_new)     # new gradient
        r_new += 0.5*epsilon*grad_new     # half-step
        return theta_new, r_new, joint_new, grad_new

    #=========================================================================
    # @functools.lru_cache(maxsize=128)
    def _BuildTree(self, theta, r, grad, nHam, log_u, v, j, epsilon, Delta_max=1000):
        # Increment the number of tree nodes counter
        self._num_tree_node += 1

        if (j == 0):     # base case
            # single leapfrog step in the direction v
            theta_prime, r_prime, joint_prime, grad_prime = self._Leapfrog_single(theta, r, grad, v*epsilon)
            nHam_prime = self._neg_Hamiltonian(joint_prime, r_prime)     # Hamiltonian eval
            n_prime = int(log_u <= nHam_prime)     # if particle is in the slice
            s_prime = int(log_u < Delta_max + nHam_prime)     # check U-turn

            # Compute the acceptance probability
            # alpha_prime = min(1, np.exp(nHam_prime - nHam))
            # written in a stable way to avoid overflow when computing
            # exp(nHam_prime - nHam) for large values of diff_Ham            
            diff_Ham = nHam_prime - nHam
            alpha_prime = 1 if diff_Ham > 0 else np.exp(diff_Ham)
            n_alpha_prime = 1

            return theta_prime, r_prime, grad_prime, theta_prime, r_prime, grad_prime, \
                theta_prime, joint_prime, grad_prime, n_prime, s_prime, alpha_prime, n_alpha_prime
 
        else: 
            # recursion: build the left/right subtrees
            theta_minus, r_minus, grad_minus, theta_plus, r_plus, grad_plus, \
            theta_prime, joint_prime, grad_prime, n_prime, s_prime, alpha_prime, n_alpha_prime = \
                self._BuildTree(theta, r, grad, nHam, log_u, v, j-1, epsilon)
            if (s_prime == 1): # do only if the stopping criteria does not verify at the first subtree
                if (v == -1):
                    theta_minus, r_minus, grad_minus, _, _, _, \
                    theta_2prime, joint_2prime, grad_2prime, n_2prime, s_2prime, alpha_2prime, n_alpha_2prime = \
                        self._BuildTree(theta_minus, r_minus, grad_minus, nHam, log_u, v, j-1, epsilon)
                else:
                    _, _, _, theta_plus, r_plus, grad_plus, \
                    theta_2prime, joint_2prime, grad_2prime, n_2prime, s_2prime, alpha_2prime, n_alpha_2prime = \
                        self._BuildTree(theta_plus, r_plus, grad_plus, nHam, log_u, v, j-1, epsilon)

                # Metropolis step
                alpha2 = n_2prime / max(1, (n_prime + n_2prime))
                if (np.random.rand() <= alpha2):
                    theta_prime = np.copy(theta_2prime)
                    joint_prime = np.copy(joint_2prime)
                    grad_prime = np.copy(grad_2prime)

                # update number of particles and stopping criterion
                alpha_prime += alpha_2prime
                n_alpha_prime += n_alpha_2prime
                #
                dtheta = theta_plus - theta_minus
                s_prime = s_2prime * int((dtheta@r_minus.T)>=0) * int((dtheta@r_plus.T)>=0)
                n_prime += n_2prime
            return theta_minus, r_minus, grad_minus, theta_plus, r_plus, grad_plus, \
                theta_prime, joint_prime, grad_prime, n_prime, s_prime, alpha_prime, n_alpha_prime
