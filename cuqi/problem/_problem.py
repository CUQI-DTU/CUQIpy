from __future__ import annotations
import numpy as np
import time
from typing import Tuple

import cuqi
from cuqi import config
from cuqi.distribution import Distribution, Gaussian, InverseGamma, LMRF, GMRF, Lognormal, Posterior, Beta, JointDistribution, Gamma, ModifiedHalfNormal, CMRF
from cuqi.implicitprior import RegularizedGaussian, RegularizedGMRF
from cuqi.density import Density
from cuqi.model import LinearModel, Model
from cuqi.likelihood import Likelihood
from cuqi.geometry import _DefaultGeometry
from cuqi.utilities import ProblemInfo
from cuqi.array import CUQIarray
import warnings
import matplotlib.pyplot as plt

from copy import copy

class BayesianProblem(object):
    """ Representation of a Bayesian inverse problem defined by any number of densities (distributions and likelihoods), e.g.

    .. math::

        \\begin{align*}
        \mathrm{density}_1 &\sim \pi_1 \\newline
        \mathrm{density}_2 &\sim \pi_2 \\newline
                           &\\vdots
        \end{align*}

    The main goal of this class is to provide fully automatic methods for computing samples or point estimates of the Bayesian problem.

    This class uses :class:`~cuqi.distribution.JointDistribution` to model the Bayesian problem,
    and to condition on observed data. We term the resulting distribution the *target distribution*.

    Parameters
    ----------
    \*densities: Density
        The densities that represent the Bayesian Problem.
        Each density is passed as comma-separated arguments.
        Can be Distribution, Likelihood etc.

    \**data: ndarray, Optional
        Any potential observed data. The data should be passed
        as keyword arguments, where the keyword is the name of the
        density that the data is associated with.
        Data can alternatively be set using the :meth:`set_data` method,
        after the problem has been initialized.

    Examples
    --------

    **Basic syntax**

    Given distributions for ``x``, ``y`` and ``z``, we can define a Bayesian problem
    and set the observed data ``y=y_data`` as follows:

    .. code-block:: python

        BP = BayesianProblem(x, y, z).set_data(y=y_data)

    **Complete example**

    Consider a Bayesian inverse problem with a Gaussian prior and a Gaussian likelihood.
    Assume that the forward model is a linear model with a known matrix
    :math:`\mathbf{A}: \mathbf{x} \mapsto \mathbf{y}` and  that we have observed data
    :math:`\mathbf{y}=\mathbf{y}^\mathrm{obs}`. The Bayesian model can be summarized as

    .. math::

        \\begin{align*}
        \mathbf{x} &\sim \mathcal{N}(\mathbf{0}, 0.1^2 \mathbf{I}) \\newline
        \mathbf{y} &\sim \mathcal{N}(\mathbf{A}\mathbf{x}, 0.05^2 \mathbf{I}).
        \end{align*}
    
    Using the :class:`BayesianProblem` class, we can define the problem, set the data and
    compute samples from the posterior distribution associated with the problem as well as
    estimates such as the Maximum Likelihood (ML) and Maximum A Posteriori (MAP).

    .. note::

        In this case, we use a forward model from the :mod:`~cuqi.testproblem` module, but any
        custom forward model can added via the :mod:`~cuqi.model` module.

    .. code-block:: python

        # Import modules
        import cuqi
        import numpy as np
        import matplotlib.pyplot as plt

        # Deterministic forward model and data (1D convolution)
        A, y_data, probInfo = cuqi.testproblem.Deconvolution1D().get_components()

        # Bayesian model
        x = cuqi.distribution.Gaussian(np.zeros(A.domain_dim), 0.1)
        y = cuqi.distribution.Gaussian(A@x, 0.05)

        # Define Bayesian problem and set data
        BP = cuqi.problem.BayesianProblem(y, x).set_data(y=y_data)

        # Compute MAP estimate
        x_MAP = BP.MAP()

        # Compute samples from posterior
        x_samples = BP.sample_posterior(1000)

        # Plot results
        x_samples.plot_ci(exact=probInfo.exactSolution)
        plt.show()
        
        # Plot difference between MAP and sample mean
        (x_MAP - x_samples.mean()).plot()
        plt.title("MAP estimate - sample mean")
        plt.show()

    Notes
    -----

    In the simplest form the Bayesian problem represents a posterior distribution defined by two densities, i.e.,
    
    .. math::

        \pi_\mathrm{posterior}(\\boldsymbol{\\theta} \mid \mathbf{y}) \propto \pi_1(\mathbf{y} \mid \\boldsymbol{\\theta}) \pi_2(\\boldsymbol{\\theta}),

    where :math:`\pi_1(\mathbf{y} \mid \\boldsymbol{\\theta})` is a :class:`~cuqi.likelihood.Likelihood` function and :math:`\pi_2(\\boldsymbol{\\theta})` is a :class:`~cuqi.distribution.Distribution`.
    In this two-density case, the joint distribution reduces to a :class:`~cuqi.distribution.Posterior` distribution.

    Most functionality is currently only implemented for this simple case.

    """

    def get_components(self) -> Tuple[Model, CUQIarray, ProblemInfo]:
        """
        Method that returns the model, the data and additional information to be used in formulating the Bayesian problem.
        
        """

        problem_info = ProblemInfo() #Instead of a dict, we use our ProblemInfo dataclass.

        for key, value in vars(problem_info).items():
            if hasattr(self, key):
                setattr(problem_info,key,vars(self)[key])

        return self.model, self.data, problem_info

    def __init__(self, *densities: Density, **data: np.ndarray):
        self._target = JointDistribution(*densities)(**data)

    def set_data(self, **kwargs) -> BayesianProblem:
        """ Set the data of the problem. This conditions the underlying joint distribution on the data. """
        if not isinstance(self._target, JointDistribution):
            raise ValueError("Unable to set data for this problem. Maybe data is already set?")
        self._target = self._target(**kwargs)
        return self

    @property
    def data(self):
        """Extract the observed data from likelihood"""
        return self.likelihood.data

    @property
    def likelihood(self) -> Likelihood:
        """The likelihood function."""
        if not isinstance(self._target, Posterior):
            raise ValueError(f"Unable to extract likelihood from this problem. Current target is: \n {self._target}")
        return self._target.likelihood

    @likelihood.setter
    def likelihood(self, likelihood):
        if not isinstance(self._target, Posterior):
            raise ValueError(f"Unable to set likelihood for this problem. Current target is: \n {self._target}")
        self._target.likelihood = likelihood

    @property
    def prior(self) -> Distribution:
        """The prior distribution"""
        if not isinstance(self._target, Posterior):
            raise ValueError(f"Unable to extract prior from this problem. Current target is: \n {self._target}")
        return self._target.prior

    @prior.setter
    def prior(self, prior):
        if not isinstance(self._target, Posterior):
            raise ValueError(f"Unable to set prior for this problem. Current target is: \n {self._target}")
        self._target.prior = prior

    @property
    def model(self) -> Model:
        """Extract the cuqi model from likelihood."""
        return self.likelihood.model

    @property
    def posterior(self) -> Posterior:
        """Create posterior distribution from likelihood and prior."""
        if not isinstance(self._target, Posterior):
            raise ValueError(f"Unable to extract posterior for this problem. Current target is: \n {self._target}")
        return self._target

    def ML(self, disp=True, x0=None) -> CUQIarray:
        """ Compute the Maximum Likelihood (ML) estimate of the posterior.
        
        Parameters
        ----------

        disp : bool
            display info messages? (True or False).

        x0 : CUQIarray or ndarray
            User-specified initial guess for the solver. Defaults to a ones vector.

        Returns
        -------
        CUQIarray
            ML estimate of the posterior. Solver info is stored in the returned CUQIarray `info` attribute.
        
        """
        if disp:
            # Print warning to user about the automatic solver selection
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!! Automatic solver selection is a work-in-progress !!!")
            print("!!!      Always validate the computed results.       !!!")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("")

        x_ML, solver_info = self._solve_max_point(self.likelihood, disp=disp, x0=x0)

        # Wrap the result in a CUQIarray and add solver info
        x_ML = cuqi.array.CUQIarray(x_ML, geometry=self.likelihood.geometry)
        x_ML.info = solver_info

        return x_ML


    def MAP(self, disp=True, x0=None) -> CUQIarray:
        """ Compute the Maximum A Posteriori (MAP) estimate of the posterior.
        
        Parameters
        ----------

        disp : bool
            display info messages? (True or False).

        x0 : CUQIarray or ndarray
            User-specified initial guess for the solver. Defaults to a ones vector.

        Returns
        -------
        CUQIarray
            MAP estimate of the posterior. Solver info is stored in the returned CUQIarray `info` attribute.
        
        """

        if disp:
            # Print warning to user about the automatic solver selection
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!! Automatic solver selection is a work-in-progress !!!")
            print("!!!      Always validate the computed results.       !!!")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("")

        if self._check_posterior(self, Gaussian, Gaussian, LinearModel, max_dim=config.MAX_DIM_INV):
            if disp: print(f"Using direct MAP of Gaussian posterior. Only works for small-scale problems with dim<={config.MAX_DIM_INV}.")
            b  = self.data
            A  = self.model.get_matrix()
            Ce = self.likelihood.distribution.cov
            x0 = self.prior.mean
            Cx = self.prior.cov

            # If Ce and Cx are scalar, make them into matrices
            if np.size(Ce)==1:
                Ce = Ce.ravel()[0]*np.eye(self.model.range_dim)
            if np.size(Cx)==1:
                Cx = Cx.ravel()[0]*np.eye(self.model.domain_dim)

            #Basic MAP estimate using closed-form expression Tarantola 2005 (3.37-3.38)
            rhs = b-A@x0
            sysm = A@Cx@A.T+Ce
            x_MAP = x0 + Cx@(A.T@np.linalg.solve(sysm,rhs))
            solver_info = {"solver": "direct"}

        else: # If no specific implementation exists, use numerical optimization.
            x_MAP, solver_info = self._solve_max_point(self.posterior, disp=disp, x0=x0)

        # Wrap the result in a CUQIarray and add solver info
        x_MAP = cuqi.array.CUQIarray(x_MAP, geometry=self.posterior.geometry)
        x_MAP.info = solver_info
        return x_MAP

    def sample_posterior(self, Ns, Nb=None, callback=None, experimental=False) -> cuqi.samples.Samples:
        """Sample the posterior. Sampler choice and tuning is handled automatically.
        
        Parameters
        ----------
        Ns : int
            Number of samples to draw.

        Nb : int or None, *Optional*
            Number of burn-in samples. If not provided, 20% of the samples will be used for burn-in.

        callback : callable, *Optional*
            If set this function will be called after every sample.
            The signature of the callback function is `callback(sample, sample_index)`,
            where `sample` is the current sample and `sample_index` is the index of the sample.
            An example is shown in demos/demo31_callback.py.
            Note: if the parameter `experimental` is set to True, the callback function should take three arguments: the sampler object, the index of the current sampling step, the total number of requested samples. The last two arguments are integers. An example of the callback function signature in the case is: `callback(sampler, sample_index, num_of_samples)`.

        experimental : bool, *Optional*
            If set to True, the sampler selection will use the samplers from the :mod:`cuqi.experimental.mcmc` module.

        Returns
        -------
        samples : cuqi.samples.Samples
            Samples from the posterior.
        
        """

        # Print warning to user about the automatic sampler selection
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! Automatic sampler selection is a work-in-progress. !!!")
        print("!!!       Always validate the computed results.        !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("")

        if experimental:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!!  Using samplers from cuqi.experimental.mcmc  !!!")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("")

        # Set up burn-in if not provided
        if Nb is None:
            Nb = int(0.2*Ns)

        # If target is a joint distribution, try Gibbs sampling
        # This is still very experimental!
        if isinstance(self._target, JointDistribution):       
            return self._sampleGibbs(Ns, Nb, callback=callback, experimental=experimental)

        # For Gaussian small-scale we can use direct sampling
        if self._check_posterior(self, Gaussian, Gaussian, LinearModel, config.MAX_DIM_INV) and not self._check_posterior(self, GMRF):
            return self._sampleMapCholesky(Ns, callback)

        # For larger-scale Gaussian we use Linear RTO. TODO: Improve checking once we have a common Gaussian class.
        elif hasattr(self.prior,"sqrtprecTimesMean") and hasattr(self.likelihood.distribution,"sqrtprec") and isinstance(self.model,LinearModel):
            return self._sampleLinearRTO(Ns, Nb, callback, experimental=experimental)

        # For LMRF we use our awesome unadjusted Laplace approximation!
        elif self._check_posterior(self, LMRF, Gaussian):
            return self._sampleUGLA(Ns, Nb, callback, experimental=experimental)

        # If we have gradients, use NUTS!
        # TODO: Fix cases where we have gradients but NUTS fails (see checks)
        elif self._check_posterior(self, must_have_gradient=True) and not self._check_posterior(self, (Beta, InverseGamma, Lognormal)):
            return self._sampleNUTS(Ns, Nb, callback, experimental=experimental)

        # For Gaussians with non-linear model we use pCN
        elif self._check_posterior(self, (Gaussian, GMRF), Gaussian):
            return self._samplepCN(Ns, Nb, callback, experimental=experimental)
        
        # For Regularized Gaussians with linear models we use RegularizedLinearRTO
        elif self._check_posterior(self, (RegularizedGaussian, RegularizedGMRF), Gaussian, LinearModel):
            return self._sampleRegularizedLinearRTO(Ns, Nb, callback, experimental=experimental)

        else:
            raise NotImplementedError(f"Automatic sampler choice is not implemented for model: {type(self.model)}, likelihood: {type(self.likelihood.distribution)} and prior: {type(self.prior)} and dim {self.prior.dim}. Manual sampler choice can be done via the 'sampler' module. Posterior distribution can be extracted via '.posterior' of any testproblem (BayesianProblem).")

    def sample_prior(self, Ns, callback=None) -> cuqi.samples.Samples:
        """ Sample the prior distribution. Sampler choice and tuning is handled automatically. """

        # Try sampling prior directly
        try:
            return self.prior.sample(Ns)
        except NotImplementedError:
            pass

        # If no direct method exists redefine posterior to one with a constant likelihood and sample from posterior
        print("Using MCMC methods from sampler module to sample prior.")
        print("Make sure enough samples are drawn for convergence.")
        print("")

        # Create a copy of self
        prior_problem = copy(self)

        # Set likelihood to constant
        model = cuqi.model.LinearModel(lambda x: 0*x, lambda y: 0*y, self.model.range_geometry, self.model.domain_geometry)
        likelihood = cuqi.distribution.Gaussian(model, 1).to_likelihood(np.zeros(self.model.range_dim)) # p(y|x)=constant
        prior_problem.likelihood = likelihood
        
        # Set up burn-in
        Nb = int(0.2*Ns)

        # Now sample prior problem
        return prior_problem.sample_posterior(Ns, Nb, callback)

    def UQ(self, Ns=1000, Nb=None, percent=95, exact=None, experimental=False) -> cuqi.samples.Samples:
        """ Run an Uncertainty Quantification (UQ) analysis on the Bayesian problem and provide a summary of the results.
        
        Parameters
        ----------
        Ns : int, *Optional*
            Number of samples to draw. Defaults to 1000.

        Nb : int, *Optional*
            Number of burn-in samples. If not provided, 20% of the samples will be used for burn-in.
        
        exact : ndarray or dict[str, ndarray], *Optional*
            Exact solution to the problem. If provided the summary will include a comparison to the exact solution.
            If a dict is provided, the keys should be the names of the variables and the values should be the exact solution for each variable.
        
        percent : float, *Optional*
            The credible interval to plot. Defaults to 95%.

        experimental : bool, *Optional*
            If set to True, the sampler selection will use the samplers from the :mod:`cuqi.experimental.mcmc` module.

        Returns
        -------
        samples : cuqi.samples.Samples
            Samples from the posterior. The samples can be used to compute further statistics and plots.
        """
        print(f"Computing {Ns} samples")
        samples = self.sample_posterior(Ns, Nb, experimental=experimental)

        print("Plotting results")
        # Gibbs case
        if isinstance(samples, dict):
            for key, value in samples.items():
                if exact is not None and key in exact:
                    self._plot_UQ_for_variable(
                        value, percent=percent, exact=exact[key])
                else:
                    self._plot_UQ_for_variable(
                        value, percent=percent, exact=None)
        # Single parameter case
        else:
            self._plot_UQ_for_variable(
                samples, percent=percent, exact=exact)

        return samples

    def _plot_UQ_for_variable(
            self, samples: cuqi.samples.Samples, percent=None, exact=None):
        """ Do a fitting UQ plot for a single variable given by samples. """
        # Potentially extract exact solution
        if exact is None and hasattr(self, 'exactSolution'):
            exact = self.exactSolution

        # Plot traces for single parameters
        if samples.shape[0] == 1:
            samples.plot_trace(exact=exact)
        else: # Else plot credible intervals
            # If plot_ci throws a NotImplementedError (likely coming from 
            # _plot_envelope method), we try to plot the CI for the parameters
            # instead and plot the mean and the variance using the function
            # representation of the samples geometry.
            try:
                samples.plot_ci(percent=percent, exact=exact)
            except NotImplementedError as nie:
                print(
                    "Unable to plot CI for samples with the underlying " +
                    f"geometry: {samples.geometry}. Plotting the CI for the " +
                    "parameters instead.")
                self._alternative_plot_UQ_for_variable(
                    samples, percent=percent, exact=exact)

    def _alternative_plot_UQ_for_variable(
            self, samples: cuqi.samples.Samples, percent=None, exact=None):
        """ Alternative visualization for UQ analysis used when plot_ci
        method fails for the given samples geometry. """
        samples.plot_ci(percent=percent, exact=exact, plot_par=True)

        plt.figure()
        samples.plot_mean()
        plt.title("Sample parameter mean converted\nto function representation")

        plt.figure()
        samples.funvals.vector.plot_mean()
        plt.title("Sample mean of function representation")

        plt.figure()
        samples.plot_variance()
        plt.title(
            "Sample parameter variance converted\nto function representation")
        
        plt.figure()
        samples.funvals.vector.plot_variance()
        plt.title("Sample variance of function representation")

    def _sampleLinearRTO(self, Ns, Nb, callback=None, experimental=False):

        if experimental:

            print("Using cuqi.experimental.mcmc LinearRTO sampler.")
            print(f"burn-in: {Nb/Ns*100:g}%")

            sampler = cuqi.experimental.mcmc.LinearRTO(self.posterior, callback=callback)

            ti = time.time()

            sampler.warmup(Nb)
            sampler.sample(Ns)
            samples = sampler.get_samples().burnthin(Nb)

            print('Elapsed time:', time.time() - ti)

        else:

            print("Using cuqi.sampler LinearRTO sampler.")
            print(f"burn-in: {Nb/Ns*100:g}%")

            # Start timing
            ti = time.time()

            # Sample
            sampler = cuqi.sampler.LinearRTO(self.posterior, callback=callback)
            samples = sampler.sample(Ns, Nb)

            # Print timing
            print('Elapsed time:', time.time() - ti)

        return samples

    def _sampleMapCholesky(self, Ns, callback=None):
        print(f"Using direct sampling of Gaussian posterior. Only works for small-scale problems with dim<={config.MAX_DIM_INV}.")
        print("No burn-in needed for direct sampling.")

        # Start timing
        ti = time.time()

        b  = self.data
        A  = self.model.get_matrix()
        Ce = self.likelihood.distribution.cov
        x0 = self.prior.mean
        Cx = self.prior.cov

        # If Ce and Cx are scalar, make them into matrices
        if np.size(Ce)==1:
            Ce = Ce.ravel()[0]*np.eye(self.model.range_dim)
        if np.size(Cx)==1:
            Cx = Cx.ravel()[0]*np.eye(self.model.domain_dim)

        # Preallocate samples
        n = self.prior.dim 
        x_s = np.zeros((n,Ns))

        x_map = self.MAP(disp=False) #Compute MAP estimate
        C = np.linalg.inv(A.T@(np.linalg.inv(Ce)@A)+np.linalg.inv(Cx))
        L = np.linalg.cholesky(C)
        for s in range(Ns):
            x_s[:,s] = x_map.parameters + L@np.random.randn(n)
            # display iterations 
            if (s % 5e2) == 0:
                print("\r",'Sample', s, '/', Ns, end="")
            
            # Callback function
            if callback is not None:
                callback(x_s[:,s], s)

        print("\r",'Sample', s+1, '/', Ns)
        print('Elapsed time:', time.time() - ti)
        
        return cuqi.samples.Samples(x_s,self.model.domain_geometry)
    
    def _sampleCWMH(self, Ns, Nb, callback=None, experimental=False):

        if experimental:

            print("Using cuqi.experimental.mcmc Component-wise Metropolis-Hastings (CWMH) sampler.")
            print(f"burn-in: {Nb/Ns*100:g}%, scale: 0.05, x0: 0.5 (vector)")

            scale = 0.05*np.ones(self.prior.dim)
            x0 = 0.5*np.ones(self.prior.dim)

            sampler = cuqi.experimental.mcmc.CWMH(self.posterior, scale, x0, callback=callback)

            ti = time.time()

            sampler.warmup(Nb)
            sampler.sample(Ns)
            x_s = sampler.get_samples().burnthin(Nb)

            print('Elapsed time:', time.time() - ti)

        else:

            print("Using cuqi.sampler Component-wise Metropolis-Hastings (CWMH) sampler (sample_adapt)")
            print(f"burn-in: {Nb/Ns*100:g}%, scale: 0.05, x0: 0.5 (vector)")

            # Dimension
            n = self.prior.dim
            
            # Set up target and proposal
            def proposal(x_t, sigma): return np.random.normal(x_t, sigma)

            # Set up sampler
            scale = 0.05*np.ones(n)
            x0 = 0.5*np.ones(n)
            MCMC = cuqi.sampler.CWMH(self.posterior, proposal, scale, x0, callback=callback)
            
            # Run sampler
            ti = time.time()
            x_s = MCMC.sample_adapt(Ns,Nb); #ToDo: Make results class
            print('Elapsed time:', time.time() - ti)
        
        return x_s

    def _samplepCN(self, Ns, Nb, callback=None, experimental=False):

        if experimental:

            print("Using cuqi.experimental.mcmc preconditioned Crank-Nicolson (pCN) sampler.")
            print(f"burn-in: {Nb/Ns*100:g}%, scale: 0.02")

            scale = 0.02

            sampler = cuqi.experimental.mcmc.PCN(self.posterior, scale, callback=callback)

            ti = time.time()

            sampler.warmup(Nb)
            sampler.sample(Ns)
            x_s = sampler.get_samples().burnthin(Nb)

            print('Elapsed time:', time.time() - ti)

        else:

            print("Using cuqi.sampler preconditioned Crank-Nicolson (pCN) sampler (sample_adapt)")
            print(f"burn-in: {Nb/Ns*100:g}%, scale: 0.02")

            scale = 0.02
            
            MCMC = cuqi.sampler.pCN(self.posterior, scale, callback=callback)      
            
            #Run sampler
            ti = time.time()
            x_s = MCMC.sample_adapt(Ns, Nb)
            print('Elapsed time:', time.time() - ti)
       
        return x_s

    def _sampleNUTS(self, Ns, Nb, callback=None, experimental=False):

        if experimental:

            print("Using cuqi.experimental.mcmc No-U-Turn (NUTS) sampler.")
            print(f"burn-in: {Nb/Ns*100:g}%")

            sampler = cuqi.experimental.mcmc.NUTS(self.posterior, callback=callback)

            ti = time.time()

            sampler.warmup(Nb)
            sampler.sample(Ns)
            x_s = sampler.get_samples().burnthin(Nb)

            print('Elapsed time:', time.time() - ti)

        else:

            print("Using cuqi.sampler No-U-Turn (NUTS) sampler")
            print(f"burn-in: {Nb/Ns*100:g}%")
            
            MCMC = cuqi.sampler.NUTS(self.posterior, callback=callback)
            
            # Run sampler
            ti = time.time()
            x_s = MCMC.sample_adapt(Ns,Nb)
            print('Elapsed time:', time.time() - ti)
        
        return x_s

    def _sampleUGLA(self, Ns, Nb, callback=None, experimental=False):

        if experimental:

            print("Using cuqi.experimental.mcmc Unadjusted Gaussian Laplace Approximation (UGLA) sampler.")
            print(f"burn-in: {Nb/Ns*100:g}%")

            sampler = cuqi.experimental.mcmc.UGLA(self.posterior, callback=callback)

            ti = time.time()

            sampler.warmup(Nb)
            sampler.sample(Ns)
            samples = sampler.get_samples().burnthin(Nb)

            print('Elapsed time:', time.time() - ti)

        else:

            print("Using cuqi.sampler UGLA sampler")
            print(f"burn-in: {Nb/Ns*100:g}%")

            # Start timing
            ti = time.time()

            # Sample
            sampler = cuqi.sampler.UGLA(self.posterior, callback=callback)
            samples = sampler.sample(Ns, Nb)

            # Print timing
            print('Elapsed time:', time.time() - ti)

        return samples
    
    def _sampleRegularizedLinearRTO(self, Ns, Nb, callback=None, experimental=False):

        if experimental:

            print("Using cuqi.experimental.mcmc Regularized LinearRTO sampler.")
            print(f"burn-in: {Nb/Ns*100:g}%")

            sampler = cuqi.experimental.mcmc.RegularizedLinearRTO(self.posterior, maxit=100, stepsize = "automatic", abstol=1e-10, callback=callback)

            ti = time.time()

            sampler.warmup(Nb)
            sampler.sample(Ns)
            samples = sampler.get_samples().burnthin(Nb)

            print('Elapsed time:', time.time() - ti)
        
        else:

            print("Using cuqi.sampler Regularized LinearRTO sampler.")
            print(f"burn-in: {Nb/Ns*100:g}%")

            # Start timing
            ti = time.time()

            # Sample
            sampler = cuqi.sampler.RegularizedLinearRTO(self.posterior, maxit=100, stepsize = "automatic", abstol=1e-10, callback=callback)
            samples = sampler.sample(Ns, Nb)

            # Print timing
            print('Elapsed time:', time.time() - ti)

        return samples

    def _solve_max_point(self, density, disp=True, x0=None):
        """ This is a helper function for point estimation of the maximum of a density (e.g. posterior or likelihood) using solver module.
        
        Parameters
        ----------
        density : Density (Distribution or Likelihood)
            The density or likelihood to compute the maximum point of the negative log.

        disp : bool
            display info messages? (True or False).
        """
        
        # Get the function to minimize (negative log-likelihood or negative log-posterior)
        def func(x): return -density.logd(x)

        # Initial value if not given
        if x0 is None:
            x0 = np.ones(self.model.domain_dim)

        # Get the gradient (if available)
        try: 
            density.gradient(x0)
            def gradfunc(x): return -density.gradient(x)
            if disp: print("Optimizing with exact gradients")
        except (NotImplementedError, AttributeError):
            gradfunc = None
            if disp: print("Optimizing with approximate gradients.") 

        # Compute point estimate
        if self._check_posterior(self, CMRF, must_have_gradient=True): # Use L-BFGS-B for CMRF prior as it has better performance for this multi-modal posterior
            if disp: print(f"Using scipy.optimize.L_BFGS_B on negative log of {density.__class__.__name__}")
            if disp: print("x0: ones vector")
            solver = cuqi.solver.ScipyLBFGSB(func, x0, gradfunc=gradfunc)
        else:
            if disp: print(f"Using scipy.optimize.minimize on negative log of {density.__class__.__name__}")
            if disp: print("x0: ones vector")
            solver = cuqi.solver.ScipyMinimizer(func, x0, gradfunc=gradfunc)

        x_MAP, solver_info = solver.solve()

        # Add info on solver choice
        solver_info["solver"] = "L-BFGS-B"

        return x_MAP, solver_info

    def _check_geometries_consistency(self, geom1, geom2, fail_msg):
        """checks geom1 and geom2 consistency . If both are of type `_DefaultGeometry` they need to be equal. If one of them is of `_DefaultGeometry` type, it will take the value of the other one. If both of them are user defined, they need to be consistent"""
        if isinstance(geom1,_DefaultGeometry):
            if isinstance(geom2,_DefaultGeometry):
                if geom1 == geom2:
                    return geom1,geom2
            else: 
                return geom2, geom2
        else:
            if isinstance(geom2,_DefaultGeometry):
                return geom1,geom1
            else:
                if geom1 == geom2:
                    return geom1,geom2
        raise Exception(fail_msg)

    @staticmethod
    def _check_posterior(posterior, prior_type=None, likelihood_type=None, model_type=None, max_dim=None, must_have_gradient=False):
        """Returns true if components of the posterior reflects the types (can be tuple of types) given as input."""
        # Prior check
        if prior_type is None:
            P = True
        else:
            P = isinstance(posterior.prior, prior_type)

        # Likelihood check
        if likelihood_type is None:
            L = True
        else:
            L = isinstance(posterior.likelihood.distribution, likelihood_type)

        # Model check
        if model_type is None:
            M = True
        else:
            M = isinstance(posterior.model, model_type)

        #Dimension check
        if max_dim is None:
            D = True
        else:
            D = posterior.model.domain_dim<=max_dim and posterior.model.range_dim<=max_dim

        # Require gradient?
        if must_have_gradient:
            try: 
                posterior.posterior.gradient(np.zeros(posterior.posterior.dim))
                G = True
            except (NotImplementedError, AttributeError):
                G = False
        else:
            G = True

        return L and P and M and D and G

    def _sampleGibbs(self, Ns, Nb, callback=None, experimental=False):
        """ This is a helper function for sampling from the posterior using Gibbs sampler. """

        if experimental:

            print("Using cuqi.experimental.mcmc HybridGibbs sampler")
            print(f"burn-in: {Nb/Ns*100:g}%")
            print("")

            # Start timing
            ti = time.time()

            # Sampling strategy
            sampling_strategy = self._determine_sampling_strategy(experimental=True)

            sampler = cuqi.experimental.mcmc.HybridGibbs(
                self._target, sampling_strategy, callback=callback)
            sampler.warmup(Nb)
            sampler.sample(Ns)
            samples = sampler.get_samples()
            # Dict with Samples objects for each parameter
            # Now apply burnthin to each value in dict
            for key, value in samples.items():
                samples[key] = value.burnthin(Nb)
            
            # Print timing
            print('Elapsed time:', time.time() - ti)

        else:

            print("Using Gibbs sampler")
            print(f"burn-in: {Nb/Ns*100:g}%")
            print("")

            if callback is not None:
                raise NotImplementedError("Callback not implemented for Gibbs sampler. It is only implemented for experimental Gibbs sampler.")

            # Start timing
            ti = time.time()

            # Sampling strategy
            sampling_strategy = self._determine_sampling_strategy()

            sampler = cuqi.sampler.Gibbs(self._target, sampling_strategy)
            samples = sampler.sample(Ns, Nb)

            # Print timing
            print('Elapsed time:', time.time() - ti)

        return samples


    def _determine_sampling_strategy(self, experimental=False):
        """ This is a helper function for determining the sampling strategy for Gibbs sampler.
        
        It is still very experimental and not very robust.
        
        """

        # We determine sampling strategy by sequentially conditioning each variable on the others.
        # We then re-use the _check_posterior method to select the best sampler for each variable.
        # In the future we may consider refactoring these methods into one more robust way of
        # determining the sampling strategy.

        # Joint distribution and parameters
        joint = self._target
        par_names = joint.get_parameter_names()

        # Go through each parameter and condition on the others, then select the best sampler
        sampling_strategy = {}
        for par_name in par_names:

            # Dict of all other parameters to condition on with ones vector as initial value
            other_params = {par_name_: np.ones(joint.get_density(par_name_).dim) for par_name_ in par_names if par_name_ != par_name}

            # Condition on all other parameters to get target conditional distribution
            cond_target = joint(**other_params)

            # If not Posterior, we cant get sampling strategy (for now)
            if not isinstance(cond_target, Posterior):
                raise NotImplementedError(f"Unable to determine sampling strategy for {par_name} with target {cond_target}")

            # Gamma or ModifiedHalfNormal prior, Gaussian or RegularizedGaussian likelihood -> Conjugate
            if self._check_posterior(cond_target, (Gamma, ModifiedHalfNormal), (Gaussian, GMRF, RegularizedGaussian, RegularizedGMRF)):
                if experimental:
                    sampling_strategy[par_name] = cuqi.experimental.mcmc.Conjugate()
                else:
                    sampling_strategy[par_name] = cuqi.sampler.Conjugate

            # Gamma prior, LMRF likelihood -> ConjugateApprox
            elif self._check_posterior(cond_target, Gamma, LMRF):
                if experimental:
                    sampling_strategy[par_name] = cuqi.experimental.mcmc.ConjugateApprox()
                else:
                    sampling_strategy[par_name] = cuqi.sampler.ConjugateApprox

            # Gaussian prior, Gaussian likelihood, Linear model -> LinearRTO
            elif self._check_posterior(cond_target, (Gaussian, GMRF), Gaussian, LinearModel):
                if experimental:
                    sampling_strategy[par_name] = cuqi.experimental.mcmc.LinearRTO()
                else:
                    sampling_strategy[par_name] = cuqi.sampler.LinearRTO

            # Implicit Regularized Gaussian prior, Gaussian likelihood, linear model -> RegularizedLinearRTO
            elif self._check_posterior(cond_target, (RegularizedGaussian, RegularizedGMRF), Gaussian, LinearModel):
                if experimental:
                    sampling_strategy[par_name] = cuqi.experimental.mcmc.RegularizedLinearRTO()
                else:
                    sampling_strategy[par_name] = cuqi.sampler.RegularizedLinearRTO

            # LMRF prior, Gaussian likelihood, Linear model -> UGLA
            elif self._check_posterior(cond_target, LMRF, Gaussian, LinearModel):
                if experimental:
                    sampling_strategy[par_name] = cuqi.experimental.mcmc.UGLA()
                else:
                    sampling_strategy[par_name] = cuqi.sampler.UGLA

            else:
                raise NotImplementedError(f"Unable to determine sampling strategy for {par_name} with target {cond_target}")

        print("Automatically determined sampling strategy:")
        for dist_name, strategy in sampling_strategy.items():
            if experimental:
                print(f"\t{dist_name}: {strategy.__class__.__name__} (mcmc.experimental)")
            else:
                print(f"\t{dist_name}: {strategy.__name__}")
        print("")

        return sampling_strategy

    def __repr__(self):
        return f"BayesianProblem with target: \n {self._target}"
