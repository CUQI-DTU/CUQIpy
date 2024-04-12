import numpy as np
import numpy as np
from cuqi.experimental.mcmc import SamplerNew
import sys
from cuqi.array import CUQIarray


# another implementation is in https://github.com/mfouesneau/NUTS
class NUTSNew(SamplerNew):
    """No-U-Turn Sampler (Hoffman and Gelman, 2014).

    Samples a distribution given its logpdf and gradient using a Hamiltonian Monte Carlo (HMC) algorithm with automatic parameter tuning.

    For more details see: See Hoffman, M. D., & Gelman, A. (2014). The no-U-turn sampler: Adaptively setting path lengths in Hamiltonian Monte Carlo. Journal of Machine Learning Research, 15, 1593-1623.

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
    def __init__(self, target, initial_point=None, max_depth=15,
                 adapt_step_size=True, opt_acc_rate=0.6, **kwargs):
        super().__init__(target, initial_point=initial_point, **kwargs)
        self.max_depth = max_depth
        self.adapt_step_size = adapt_step_size
        self.opt_acc_rate = opt_acc_rate

        # NUTS run diagnostic
        # number of tree nodes created each NUTS iteration
        self._num_tree_node = 0
        # Create lists to store NUTS run diagnostics
        self._create_run_diagnostic_attributes()
        self._acc = [None]
        self.current_point = self.initial_point

        # Fixed parameters that do not change during the run
        self._gamma, self._t_0, self._kappa = 0.05, 10, 0.75 # kappa in (0.5, 1]
        self._delta = self.opt_acc_rate # https://mc-stan.org/docs/2_18/reference-manual/hmc-algorithm-parameters.html

        # Parameters that change during the run
        self._step_size = []
        self._epsilon = None
        self._epsilon_bar = None
        
    #=========================================================================
    #================== Implement methods required by SamplerNew =============
    #=========================================================================
    def validate_target(self):
        pass #TODO: target needs to have logpdf and gradient methods

    def step(self):
        self.current_point = self.current_point.to_numpy() if hasattr(self.current_point, 'to_numpy') else self.current_point
        self._joint = self._joint.to_numpy() if hasattr(self._joint, 'to_numpy') else self._joint
        self._grad = self._grad.to_numpy() if hasattr(self._grad, 'to_numpy') else self._grad

        # reset number of tree nodes for each iteration
        self._num_tree_node = 0

        self._step_size.append(self._epsilon)

        theta_k, joint_k = self.current_point.copy(), self._joint # initial position (parameters)
        grad = self._grad.copy() # initial gradient
        r_k = self._Kfun(1, 'sample') # resample momentum vector
        Ham = joint_k - self._Kfun(r_k, 'eval') # Hamiltonian

        # slice variable
        log_u = Ham - np.random.exponential(1, size=1) # u = np.log(np.random.uniform(0, np.exp(H)))

        # initialization
        j, s, n = 0, 1, 1
        self.current_point, self._joint = theta_k, joint_k
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
                    self._BuildTree(theta_minus, r_minus, grad_minus, Ham, log_u, v, j, self._epsilon)
            else:
                _, _, _, theta_plus, r_plus, grad_plus, \
                theta_prime, joint_prime, grad_prime, n_prime, s_prime, alpha, n_alpha = \
                    self._BuildTree(theta_plus, r_plus, grad_plus, Ham, log_u, v, j, self._epsilon)

            # Metropolis step
            alpha2 = min(1, (n_prime/n)) #min(0, np.log(n_p) - np.log(n))
            if (s_prime == 1) and (np.random.rand() <= alpha2):
                self.current_point = theta_prime
                self._joint = joint_prime
                self._grad = np.copy(grad_prime)

            # update number of particles, tree level, and stopping criterion
            n += n_prime
            dtheta = theta_plus - theta_minus
            s = s_prime * int((dtheta @ r_minus.T) >= 0) * int((dtheta @ r_plus.T) >= 0)
            j += 1
            self._alpha = alpha
            self._n_alpha = n_alpha

        # update run diagnostic attributes
        self._update_run_diagnostic_attributes(
            len(self._step_size), self._num_tree_node, self._epsilon, self._epsilon_bar)
        
        self._epsilon = self._epsilon_bar 
        if np.isnan(self._joint):
            raise NameError('NaN potential func')

    def tune(self, skip_len, update_count):
        # adapt epsilon during burn-in using dual averaging
        k = update_count+1

        eta1 = 1/(k + self._t_0)
        self._H_bar = (1-eta1)*self._H_bar + eta1*(self._delta - (self._alpha/self._n_alpha))
        self._H_bar = self._H_bar.to_numpy() if isinstance(self._H_bar, CUQIarray) else self._H_bar
        self._epsilon = np.exp(self._mu - (np.sqrt(k)/self._gamma)*self._H_bar)
        eta = k**(-self._kappa)
        self._epsilon = self._epsilon.to_numpy() if isinstance(self._epsilon, CUQIarray) else self._epsilon
        self._epsilon_bar = np.exp(eta*np.log(self._epsilon) + (1-eta)*np.log(self._epsilon_bar))

    def get_state(self):
        pass

    def set_state(self, state):
        pass

    def _pre_warmup(self):
        # Reset run diagnostic attributes
        self._reset_run_diagnostic_attributes()

        # parameters that change during the run
        self._epsilon_bar, self._H_bar = 1, 0

        # parameters dual averaging
        self._epsilon = self._FindGoodEpsilon()

        # Parameter mu, does not change during the run
        self._mu = np.log(10*self._epsilon)

    def _pre_sample(self):
        self._joint, self._grad = self._nuts_target(self.current_point)

        if self._epsilon is None:
            self._epsilon = self.adapt_step_size
            self._epsilon_bar = self.adapt_step_size
 
    #=========================================================================
    def _nuts_target(self, x): # returns logposterior tuple evaluation-gradient
        return self.target.logd(x), self.target.gradient(x)

    #=========================================================================
    # auxiliary standard Gaussian PDF: kinetic energy function
    # d_log_2pi = d*np.log(2*np.pi)
    def _Kfun(self, r, flag):
        if flag == 'eval': # evaluate
            return 0.5*(r.T @ r) #+ d_log_2pi 
        if flag == 'sample': # sample
            return np.random.standard_normal(size=self.dim)

    #=========================================================================
    def _FindGoodEpsilon(self, epsilon=1):
        theta = self.current_point
        self._joint, self._grad = self._nuts_target(theta)
        joint = self._joint
        grad = self._grad

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
    def _Leapfrog(self, theta_old, r_old, grad_old, epsilon):
        # symplectic integrator: trajectories preserve phase space volumen
        r_new = r_old + 0.5*epsilon*grad_old     # half-step
        theta_new = theta_old + epsilon*r_new     # full-step
        joint_new, grad_new = self._nuts_target(theta_new)     # new gradient
        r_new += 0.5*epsilon*grad_new     # half-step
        return theta_new, r_new, joint_new, grad_new

    #=========================================================================
    # @functools.lru_cache(maxsize=128)
    def _BuildTree(self, theta, r, grad, Ham, log_u, v, j, epsilon, Delta_max=1000):
        # Increment the number of tree nodes counter
        self._num_tree_node += 1

        if (j == 0):     # base case
            # single leapfrog step in the direction v
            theta_prime, r_prime, joint_prime, grad_prime = self._Leapfrog(theta, r, grad, v*epsilon)
            Ham_prime = joint_prime - self._Kfun(r_prime, 'eval')     # Hamiltonian eval
            n_prime = int(log_u <= Ham_prime)     # if particle is in the slice
            s_prime = int(log_u < Delta_max + Ham_prime)     # check U-turn
            #
            diff_Ham = Ham_prime - Ham

            # Compute the acceptance probability
            # alpha_prime = min(1, np.exp(diff_Ham))
            # written in a stable way to avoid overflow when computing
            # exp(diff_Ham) for large values of diff_Ham
            alpha_prime = 1 if diff_Ham > 0 else np.exp(diff_Ham)
            n_alpha_prime = 1
            #
            theta_minus, theta_plus = theta_prime, theta_prime
            r_minus, r_plus = r_prime, r_prime
            grad_minus, grad_plus = grad_prime, grad_prime
        else: 
            # recursion: build the left/right subtrees
            theta_minus, r_minus, grad_minus, theta_plus, r_plus, grad_plus, \
            theta_prime, joint_prime, grad_prime, n_prime, s_prime, alpha_prime, n_alpha_prime = \
                self._BuildTree(theta, r, grad, Ham, log_u, v, j-1, epsilon)
            if (s_prime == 1): # do only if the stopping criteria does not verify at the first subtree
                if (v == -1):
                    theta_minus, r_minus, grad_minus, _, _, _, \
                    theta_2prime, joint_2prime, grad_2prime, n_2prime, s_2prime, alpha_2prime, n_alpha_2prime = \
                        self._BuildTree(theta_minus, r_minus, grad_minus, Ham, log_u, v, j-1, epsilon)
                else:
                    _, _, _, theta_plus, r_plus, grad_plus, \
                    theta_2prime, joint_2prime, grad_2prime, n_2prime, s_2prime, alpha_2prime, n_alpha_2prime = \
                        self._BuildTree(theta_plus, r_plus, grad_plus, Ham, log_u, v, j-1, epsilon)

                # Metropolis step
                alpha2 = n_2prime / max(1, (n_prime + n_2prime))
                if (np.random.rand() <= alpha2):
                    theta_prime = np.copy(theta_2prime)
                    joint_prime = np.copy(joint_2prime)
                    grad_prime = np.copy(grad_2prime)

                # update number of particles and stopping criterion
                alpha_prime += alpha_2prime
                n_alpha_prime += n_alpha_2prime
                dtheta = theta_plus - theta_minus
                s_prime = s_2prime * int((dtheta@r_minus.T)>=0) * int((dtheta@r_plus.T)>=0)
                n_prime += n_2prime
        return theta_minus, r_minus, grad_minus, theta_plus, r_plus, grad_plus, \
                theta_prime, joint_prime, grad_prime, n_prime, s_prime, alpha_prime, n_alpha_prime

    #=========================================================================
    #======================== Diagnostic methods =============================
    #=========================================================================

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

    def _print_progress(self,s,Ns):
        """Prints sampling progress"""
        if Ns > 2:
            if (s % (max(Ns//100,1))) == 0:
                msg = f'Sample {s} / {Ns}'
                sys.stdout.write('\r'+msg)
            if s==Ns:
                msg = f'Sample {s} / {Ns}'
                sys.stdout.write('\r'+msg+'\n')
