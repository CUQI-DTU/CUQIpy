import numpy as np
import numpy as np
from cuqi.experimental.mcmc import Sampler
from cuqi.array import CUQIarray
from numbers import Number

class NUTS(Sampler):
    """No-U-Turn Sampler (Hoffman and Gelman, 2014).

    Samples a distribution given its logpdf and gradient using a Hamiltonian
    Monte Carlo (HMC) algorithm with automatic parameter tuning.

    For more details see: See Hoffman, M. D., & Gelman, A. (2014). The no-U-turn
    sampler: Adaptively setting path lengths in Hamiltonian Monte Carlo. Journal
    of Machine Learning Research, 15, 1593-1623.

    Parameters
    ----------
    target : `cuqi.distribution.Distribution`
        The target distribution to sample. Must have logpdf and gradient method.
        Custom logpdfs and gradients are supported by using a
        :class:`cuqi.distribution.UserDefinedDistribution`.
    
    initial_point : ndarray
        Initial parameters. *Optional*. If not provided, the initial point is 
        an array of ones.

    max_depth : int
        Maximum depth of the tree >=0 and the default is 15.

    step_size : None or float
        If step_size is provided (as positive float), it will be used as initial
        step size. If None, the step size will be estimated by the sampler.

    opt_acc_rate : float
        The optimal acceptance rate to reach if using adaptive step size.
        Suggested values are 0.6 (default) or 0.8 (as in stan). In principle,
        opt_acc_rate should be in (0, 1), however, choosing a value that is very
        close to 1 or 0 might lead to poor performance of the sampler.

    callback : callable, optional
        A function that will be called after each sampling step. It can be useful for monitoring the sampler during sampling.
        The function should take three arguments: the sampler object, the index of the current sampling step, the total number of requested samples. The last two arguments are integers. An example of the callback function signature is: `callback(sampler, sample_index, num_of_samples)`.

    Example
    -------
    .. code-block:: python

        # Import cuqi
        import cuqi

        # Define a target distribution
        tp = cuqi.testproblem.WangCubic()
        target = tp.posterior

        # Set up sampler
        sampler = cuqi.experimental.mcmc.NUTS(target)

        # Sample
        sampler.warmup(5000)
        sampler.sample(10000)

        # Get samples
        samples = sampler.get_samples()

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
        # only used after adaptation).
        sampler.epsilon_bar_list

    """

    _STATE_KEYS = Sampler._STATE_KEYS.union({'_epsilon', '_epsilon_bar',
                                                '_H_bar',
                                                'current_target_logd',
                                                'current_target_grad',
                                                'max_depth'})

    _HISTORY_KEYS = Sampler._HISTORY_KEYS.union({'num_tree_node_list',
                                                    'epsilon_list',
                                                    'epsilon_bar_list'})

    def __init__(self, target=None, initial_point=None, max_depth=None,
                 step_size=None, opt_acc_rate=0.6, **kwargs):
        super().__init__(target, initial_point=initial_point, **kwargs)

        # Assign parameters as attributes
        self.max_depth = max_depth
        self.step_size = step_size
        self.opt_acc_rate = opt_acc_rate


    def _initialize(self):

        self._current_alpha_ratio = np.nan # Current alpha ratio will be set to some
                                           # value (other than np.nan) before 
                                           # being used

        self.current_target_logd, self.current_target_grad = self._nuts_target(self.current_point)

        # Parameters dual averaging
        # Initialize epsilon and epsilon_bar
        # epsilon is the step size used in the current iteration
        # after warm up and one sampling step, epsilon is updated
        # to epsilon_bar for the remaining sampling steps.
        if self.step_size is None:
            self._epsilon = self._FindGoodEpsilon()
        else:
            self._epsilon = self.step_size
        self._epsilon_bar = "unset"

        # Parameter mu, does not change during the run
        self._mu = np.log(10*self._epsilon)

        self._H_bar = 0

        # NUTS run diagnostic:
        # number of tree nodes created each NUTS iteration
        self._num_tree_node = 0

        # Create lists to store NUTS run diagnostics
        self._create_run_diagnostic_attributes()

    #=========================================================================
    #============================== Properties ===============================
    #=========================================================================
    @property
    def max_depth(self):
        return self._max_depth

    @max_depth.setter
    def max_depth(self, value):
        if value is None:
            value = 15 # default value
        if not isinstance(value, int):
            raise TypeError('max_depth must be an integer.')
        if value < 0:
            raise ValueError('max_depth must be >= 0.')
        self._max_depth = value

    @property
    def step_size(self):
        return self._step_size
    
    @step_size.setter
    def step_size(self, value):
        if value is None:
            pass # NUTS will adapt the step size

        # step_size must be a positive float, raise error otherwise
        elif isinstance(value, bool)\
            or not isinstance(value, Number)\
            or value <= 0:
            raise TypeError('step_size must be a positive float or None.')
        self._step_size = value

    @property
    def opt_acc_rate(self):
        return self._opt_acc_rate
    
    @opt_acc_rate.setter
    def opt_acc_rate(self, value):
        if not isinstance(value, Number) or value <= 0 or value >= 1:
            raise ValueError('opt_acc_rate must be a float in (0, 1).')
        self._opt_acc_rate = value

    #=========================================================================
    #================== Implement methods required by Sampler =============
    #=========================================================================
    def validate_target(self):
        # Check if the target has logd and gradient methods
        try:
            current_target_logd, current_target_grad =\
            self._nuts_target(np.ones(self.dim))
        except:
            raise ValueError('Target must have logd and gradient methods.')

    def reinitialize(self):
        # Call the parent reset method
        super().reinitialize()
        # Reset NUTS run diagnostic attributes
        self._reset_run_diagnostic_attributes()

    def step(self):
        if isinstance(self._epsilon_bar, str) and self._epsilon_bar == "unset":
            self._epsilon_bar = self._epsilon

        # Convert current_point, logd, and grad to numpy arrays
        # if they are CUQIarray objects
        if isinstance(self.current_point, CUQIarray):
            self.current_point = self.current_point.to_numpy() 
        if isinstance(self.current_target_logd, CUQIarray):
            self.current_target_logd = self.current_target_logd.to_numpy()
        if isinstance(self.current_target_grad, CUQIarray):
            self.current_target_grad = self.current_target_grad.to_numpy()

        # reset number of tree nodes for each iteration
        self._num_tree_node = 0

        # copy current point, logd, and grad in local variables
        point_k = self.current_point # initial position (parameters)
        logd_k = self.current_target_logd
        grad_k = self.current_target_grad # initial gradient
        
        # compute r_k and Hamiltonian
        r_k = self._Kfun(1, 'sample') # resample momentum vector
        Ham = logd_k - self._Kfun(r_k, 'eval') # Hamiltonian

        # slice variable
        log_u = Ham - np.random.exponential(1, size=1)

        # initialization
        j, s, n = 0, 1, 1
        point_minus, point_plus = point_k.copy(), point_k.copy()
        grad_minus, grad_plus = grad_k.copy(), grad_k.copy()
        r_minus, r_plus = r_k.copy(), r_k.copy()

        # run NUTS
        acc = 0
        while (s == 1) and (j <= self.max_depth):
            # sample a direction
            v = int(2*(np.random.rand() < 0.5)-1)

            # build tree: doubling procedure
            if (v == -1):
                point_minus, r_minus, grad_minus, _, _, _, \
                    point_prime, logd_prime, grad_prime,\
                        n_prime, s_prime, alpha, n_alpha = \
                            self._BuildTree(point_minus, r_minus, grad_minus,
                                            Ham, log_u, v, j, self._epsilon)
            else:
                _, _, _, point_plus, r_plus, grad_plus, \
                    point_prime, logd_prime, grad_prime,\
                        n_prime, s_prime, alpha, n_alpha = \
                            self._BuildTree(point_plus, r_plus, grad_plus,
                                            Ham, log_u, v, j, self._epsilon)

            # Metropolis step
            alpha2 = min(1, (n_prime/n)) #min(0, np.log(n_p) - np.log(n))
            if (s_prime == 1) and \
                (np.random.rand() <= alpha2) and \
                (not np.isnan(logd_prime)) and \
                (not np.isinf(logd_prime)):
                self.current_point = point_prime.copy()
                # copy if array, else assign if scalar
                self.current_target_logd = (
                        logd_prime.copy()
                        if isinstance(logd_prime, np.ndarray)
                        else logd_prime
                    )
                self.current_target_grad = grad_prime.copy()
                acc = 1


            # update number of particles, tree level, and stopping criterion
            n += n_prime
            dpoints = point_plus - point_minus
            s = s_prime *\
                int((dpoints @ r_minus.T) >= 0) * int((dpoints @ r_plus.T) >= 0)
            j += 1
            self._current_alpha_ratio = alpha/n_alpha

        # update run diagnostic attributes
        self._update_run_diagnostic_attributes(
            self._num_tree_node, self._epsilon, self._epsilon_bar)
        
        self._epsilon = self._epsilon_bar 
        if np.isnan(self.current_target_logd):
            raise NameError('NaN potential func')

        return acc

    def tune(self, skip_len, update_count):
        """ adapt epsilon during burn-in using dual averaging"""
        if isinstance(self._epsilon_bar, str) and self._epsilon_bar == "unset":
            self._epsilon_bar = 1

        k = update_count+1

        # Fixed parameters that do not change during the run
        gamma, t_0, kappa = 0.05, 10, 0.75 # kappa in (0.5, 1]

        eta1 = 1/(k + t_0)
        self._H_bar = (1-eta1)*self._H_bar +\
            eta1*(self.opt_acc_rate - (self._current_alpha_ratio))
        self._epsilon = np.exp(self._mu - (np.sqrt(k)/gamma)*self._H_bar)
        eta = k**(-kappa)
        self._epsilon_bar =\
            np.exp(eta*np.log(self._epsilon) +(1-eta)*np.log(self._epsilon_bar))

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
        point_k = self.current_point
        self.current_target_logd, self.current_target_grad = self._nuts_target(
            point_k)
        logd = self.current_target_logd
        grad = self.current_target_grad

        r = self._Kfun(1, 'sample')    # resample a momentum
        Ham = logd - self._Kfun(r, 'eval')     # initial Hamiltonian
        _, r_prime, logd_prime, grad_prime = self._Leapfrog(
            point_k, r, grad, epsilon)

        # trick to make sure the step is not huge, leading to infinite values of
        # the likelihood
        k = 1
        while np.isinf(logd_prime) or np.isinf(grad_prime).any():
            k *= 0.5
            _, r_prime, logd_prime, grad_prime = self._Leapfrog(
                point_k, r, grad, epsilon*k)
        epsilon = 0.5*k*epsilon

        # doubles/halves the value of epsilon until the accprob of the Langevin
        # proposal crosses 0.5
        Ham_prime = logd_prime - self._Kfun(r_prime, 'eval')
        log_ratio = Ham_prime - Ham
        a = 1 if log_ratio > np.log(0.5) else -1
        while (a*log_ratio > -a*np.log(2)):
            epsilon = (2**a)*epsilon
            _, r_prime, logd_prime, _ = self._Leapfrog(
                point_k, r, grad, epsilon)
            Ham_prime = logd_prime - self._Kfun(r_prime, 'eval')
            log_ratio = Ham_prime - Ham
        return epsilon

    #=========================================================================
    def _Leapfrog(self, point_old, r_old, grad_old, epsilon):
        # symplectic integrator: trajectories preserve phase space volumen
        r_new = r_old + 0.5*epsilon*grad_old     # half-step
        point_new = point_old + epsilon*r_new     # full-step
        logd_new, grad_new = self._nuts_target(point_new)     # new gradient
        r_new += 0.5*epsilon*grad_new     # half-step
        return point_new, r_new, logd_new, grad_new

    #=========================================================================
    def _BuildTree(
            self, point_k, r, grad, Ham, log_u, v, j, epsilon, Delta_max=1000):
        # Increment the number of tree nodes counter
        self._num_tree_node += 1

        if (j == 0):     # base case
            # single leapfrog step in the direction v
            point_prime, r_prime, logd_prime, grad_prime = self._Leapfrog(
                point_k, r, grad, v*epsilon)
            Ham_prime = logd_prime - self._Kfun(r_prime, 'eval') # Hamiltonian
                                                                 # eval
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
            point_minus, point_plus = point_prime, point_prime
            r_minus, r_plus = r_prime, r_prime
            grad_minus, grad_plus = grad_prime, grad_prime
        else: 
            # recursion: build the left/right subtrees
            point_minus, r_minus, grad_minus, point_plus, r_plus, grad_plus, \
                point_prime, logd_prime, grad_prime,\
                    n_prime, s_prime, alpha_prime, n_alpha_prime = \
                        self._BuildTree(point_k, r, grad,
                                        Ham, log_u, v, j-1, epsilon)
            if (s_prime == 1): # do only if the stopping criteria does not 
                               # verify at the first subtree
                if (v == -1):
                    point_minus, r_minus, grad_minus, _, _, _, \
                        point_2prime, logd_2prime, grad_2prime,\
                            n_2prime, s_2prime, alpha_2prime, n_alpha_2prime = \
                                self._BuildTree(point_minus, r_minus, grad_minus,
                                                Ham, log_u, v, j-1, epsilon)
                else:
                    _, _, _, point_plus, r_plus, grad_plus, \
                        point_2prime, logd_2prime, grad_2prime,\
                            n_2prime, s_2prime, alpha_2prime, n_alpha_2prime = \
                                self._BuildTree(point_plus, r_plus, grad_plus,
                                                Ham, log_u, v, j-1, epsilon)

                # Metropolis step
                alpha2 = n_2prime / max(1, (n_prime + n_2prime))
                if (np.random.rand() <= alpha2):
                    point_prime = point_2prime.copy()
                    # copy if array, else assign if scalar
                    logd_prime = (
                        logd_2prime.copy()
                        if isinstance(logd_2prime, np.ndarray)
                        else logd_2prime
                    )
                    grad_prime = grad_2prime.copy()

                # update number of particles and stopping criterion
                alpha_prime += alpha_2prime
                n_alpha_prime += n_alpha_2prime
                dpoints = point_plus - point_minus
                s_prime = s_2prime *\
                    int((dpoints@r_minus.T)>=0) * int((dpoints@r_plus.T)>=0)
                n_prime += n_2prime

        return point_minus, r_minus, grad_minus, point_plus, r_plus, grad_plus,\
            point_prime, logd_prime, grad_prime,\
                n_prime, s_prime, alpha_prime, n_alpha_prime

    #=========================================================================
    #======================== Diagnostic methods =============================
    #=========================================================================

    def _create_run_diagnostic_attributes(self):
        """A method to create attributes to store NUTS run diagnostic."""
        self._reset_run_diagnostic_attributes()

    def _reset_run_diagnostic_attributes(self):
        """A method to reset attributes to store NUTS run diagnostic."""
        # List to store number of tree nodes created each NUTS iteration
        self.num_tree_node_list = []
        # List of step size used in each NUTS iteration 
        self.epsilon_list = []
        # List of burn-in step size suggestion during adaptation 
        # only used when adaptation is done
        # remains fixed after adaptation (after burn-in)
        self.epsilon_bar_list = []

    def _update_run_diagnostic_attributes(self, n_tree, eps, eps_bar):
        """A method to update attributes to store NUTS run diagnostic."""
        # Store the number of tree nodes created in iteration k
        self.num_tree_node_list.append(n_tree)
        # Store the step size used in iteration k
        self.epsilon_list.append(eps)
        # Store the step size suggestion during adaptation in iteration k
        self.epsilon_bar_list.append(eps_bar)