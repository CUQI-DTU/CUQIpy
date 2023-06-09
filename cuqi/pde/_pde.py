from abc import ABC, abstractmethod
import scipy
from inspect import getsource
from scipy.interpolate import interp1d
import numpy as np

class PDE(ABC):
    """
    Parametrized PDE abstract base class

    Parameters
    -----------   
    PDE_form : callable function
        Callable function which returns a tuple of the needed PDE components (expected components are explained in the subclasses) 

    observation_map: a function handle
        A function that takes the PDE solution as input and the returns the observed solution. e.g. `observation_map=lambda u: u**2` or `observation_map=lambda u: u[0]`

    grid_sol: np.ndarray
        The grid on which solution is defined

    grid_obs: np.ndarray
        The grid on which the observed solution should be interpolated (currently only supported for 1D problems).  
    """

    def __init__(self, PDE_form, grid_sol=None, grid_obs=None, observation_map=None):
        self.PDE_form = PDE_form
        self.grid_sol = grid_sol
        self.grid_obs = grid_obs
        self.observation_map = observation_map

    @abstractmethod
    def assemble(self,parameter):
        pass

    @abstractmethod
    def solve(self):
        pass

    @abstractmethod
    def observe(self,solution):
        pass

    def __repr__(self) -> str:
        msg = "CUQI {}.".format(self.__class__.__name__)
        if hasattr(self,"PDE_form"):
            msg+="\nPDE form expression:\n{}".format(getsource(self.PDE_form))
        return msg
        
    @staticmethod
    def _compare_grid(grid1,grid2):
        """Compares two grids and returns if they are equal"""

        # If one of the grids are none, we assume they are equal
        if grid1 is None or grid2 is None:
            return True

        m, n = len(grid1), len(grid2)
        if (m == n):            
            equal_arrays = (grid1 == grid2).all()
        else:
            equal_arrays = False

        return equal_arrays

    @property
    def grid_sol(self):
        if hasattr(self,"_grid_sol"):
            return self._grid_sol
        else:
            return None

    @grid_sol.setter
    def grid_sol(self,value):
        self._grids_equal = self._compare_grid(value,self.grid_obs)
        self._grid_sol = value

    @property
    def grid_obs(self):
        if hasattr(self,"_grid_obs"):
            return self._grid_obs
        else:
            return None

    @grid_obs.setter
    def grid_obs(self,value):
        if value is None:
            value = self.grid_sol
        self._grids_equal = self._compare_grid(value,self.grid_sol)
        self._grid_obs = value

    @property
    def grids_equal(self):
        return self._grids_equal


class LinearPDE(PDE):
    """
    Parametrized Linear PDE base class

    Parameters
    -----------   
    PDE_form : callable function
        Callable function which returns a tuple of the needed PDE components (expected components are explained in the subclasses) 

    linalg_solve: lambda function or function handle
        linear system solver function to solve the arising linear system with the signature :meth:`x, val1, val2, ...=linalg_solve(A,b,**linalg_solve_kwargs)` where A is the linear operator and b is the right hand side. `linalg_solve_kwargs` is any keywords arguments that the function :meth:`linalg_solve` can take. x is the solution of A*x=b of type `numpy.ndarray`. val1, val2, etc. are optional and can be a one or more values the solver return, e.g. information and number of iterations (for iterative solvers). If linalg_solve is None, :meth:`scipy.linalg.solve` will be used. 

    linalg_solve_kwargs: a dictionary 
        A dictionary of the keywords arguments that linalg_solve can take.

    kwargs: 
        See :class:`~cuqi.pde.PDE` for the remaining keyword arguments. 
    """

    def __init__(self, PDE_form, linalg_solve=None, linalg_solve_kwargs=None, **kwargs):
        super().__init__(PDE_form, **kwargs)

        if linalg_solve is None:
            linalg_solve = scipy.linalg.solve
        if linalg_solve_kwargs is None:
            linalg_solve_kwargs = {}

        self._linalg_solve = linalg_solve
        self._linalg_solve_kwargs = linalg_solve_kwargs

    def _solve_linear_system(self, A, b, linalg_solve, kwargs):
        """Helper function that solves the linear system `A*x=b` using the provided solve method `linalg_solve` and its keyword arguments `kwargs`. It then returns the output in the format: `solution`, `info`"""
        returned_values = linalg_solve(A, b, **kwargs)
        if isinstance(returned_values, tuple):
            solution = returned_values[0]
            info = returned_values[1:]
        else:
            solution = returned_values
            info = None

        return solution, info

class SteadyStateLinearPDE(LinearPDE):
    """Linear steady state PDE.
    
    Parameters
    -----------   
    PDE_form : callable function
        Callable function with signature `PDE_form(parameter)` where `parameter` is the Bayesian parameter. The function returns a tuple with the discretized differential operator A and right-hand-side b. The types of A and b are determined by what the method :meth:`linalg_solve` accepts as first and second parameters, respectively. 

    kwargs: 
        See :class:`~cuqi.pde.LinearPDE` for the remaining keyword arguments. 

    Example
    -------- 
    See demo demos/demo24_fwd_poisson.py for an illustration on how to use SteadyStateLinearPDE with varying solver choices. And demos demos/demo25_fwd_poisson_2D.py and demos/demo26_fwd_poisson_mixedBC.py for examples with mixed (Dirichlet and Neumann) boundary conditions problems. demos/demo25_fwd_poisson_2D.py also illustrates how to observe on a specific boundary, for example.
    """

    def __init__(self, PDE_form, **kwargs):
        super().__init__(PDE_form, **kwargs)

    def assemble(self, parameter):
        """Assembles differential operator and rhs according to PDE_form"""
        self.diff_op, self.rhs = self.PDE_form(parameter)

    def solve(self):
        """Solve the PDE and returns the solution and an information variable `info` which is a tuple of all variables returned by the function `linalg_solve` after the solution."""
        if not hasattr(self, "diff_op") or not hasattr(self, "rhs"):
            raise Exception("PDE is not assembled.")

        return self._solve_linear_system(self.diff_op, self.rhs, self._linalg_solve, self._linalg_solve_kwargs)


    def observe(self, solution):
            
        if self.grids_equal:
            solution_obs = solution
        else:
            solution_obs = interp1d(self.grid_sol, solution, kind='quadratic')(self.grid_obs)

        if self.observation_map is not None:
            solution_obs = self.observation_map(solution_obs)
                
        return solution_obs
        
class TimeDependentLinearPDE(LinearPDE):
    """Time Dependent Linear PDE with fixed time stepping using Euler method (backward or forward).
    
    Parameters
    -----------   
    PDE_form : callable function
        Callable function with signature `PDE_form(parameter, t)` where `parameter` is the Bayesian parameter and `t` is the time at which the PDE form is evaluated. The function returns a tuple of (`differential_operator`, `source_term`, `initial_condition`) where `differential_operator` is the linear operator at time `t`, `source_term` is the source term at time `t`, and `initial_condition` is the initial condition. The types of `differential_operator` and `source_term` are determined by what the method :meth:`linalg_solve` accepts as linear operator and right-hand side, respectively. The type of `initial_condition` should be the same type as the solution returned by :meth:`linalg_solve`.

    time_steps : ndarray 
        An array of the discretized times corresponding to the time steps that starts with the initial time and ends with the final time
        
    time_obs : array_like or str
        If passed as an array_like, it is an array of the times at which the solution is observed. If passed as a string it can be set to `final` to observe at the final time step, or `all` to observe at all time steps. Default is `final`.

    method: str
        Time stepping method. Currently two options are available `forward_euler` and  `backward_euler`.

    kwargs: 
        See :class:`~cuqi.pde.LinearPDE` for the remaining keyword arguments 
 
    Example
    -----------  
    See demos/demo34_TimeDependentLinearPDE.py for 1D heat and 1D wave equations.
    """

    def __init__(self, PDE_form, time_steps, time_obs='final', method='forward_euler', **kwargs):
        super().__init__(PDE_form, **kwargs)

        self.time_steps = time_steps
        self.method = method

        # Set time_obs
        if time_obs is None:
            raise ValueError("time_obs cannot be None")
        elif isinstance(time_obs, str):
            if time_obs.lower() == 'final':
                time_obs = time_steps[-1:]
            elif time_obs.lower() == 'all':
                time_obs = time_steps
            else:
                raise ValueError("if time_obs is a string, it can only be set "
                                 +"to `final` or `all`")
        self._time_obs = time_obs

    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, value):
        if value.lower() != 'forward_euler' and value.lower() != 'backward_euler':
            raise ValueError(
                "method can be set to either `forward_euler` or `backward_euler`")
        self._method = value

    def assemble(self, parameter):
        """Assemble PDE"""
        self._parameter = parameter

    def assemble_step(self, t):
        """Assemble time step at time t"""
        self.diff_op, self.rhs, self.initial_condition = self.PDE_form(self._parameter, t)

    def solve(self):
        """Solve PDE by time-stepping"""
        # initialize time-dependent solution
        self.assemble_step(self.time_steps[0])
        u = np.empty((len(self.initial_condition), len(self.time_steps)))
        u[:, 0] = self.initial_condition

        if self.method == 'forward_euler':
            for idx, t in enumerate(self.time_steps[:-1]):
                dt = self.time_steps[idx+1] - t
                self.assemble_step(t)
                u_pre = u[:, idx]
                u[:, idx+1] = (dt*self.diff_op + np.eye(len(u_pre)))@u_pre + dt*self.rhs  # from u at time t, gives u at t+dt
            info = None

        if self.method == 'backward_euler':
            for idx, t in enumerate(self.time_steps[1:]):
                dt = t - self.time_steps[idx]
                self.assemble_step(t)
                u_pre = u[:, idx]
                A = np.eye(len(u_pre)) - dt*self.diff_op
                # from u at time t-dt, gives u at t
                u[:, idx+1], info = self._solve_linear_system(
                    A, u_pre + dt*self.rhs, self._linalg_solve, self._linalg_solve_kwargs)

        return u, info

    def observe(self, solution):

        # If observation grid is the same as solution grid and observation time
        # is the final time step then no need to interpolate
        if self.grids_equal and np.all(self.time_steps[-1:] == self._time_obs):
            solution_obs = solution[..., -1]

        # Interpolate solution in time and space to the observation
        # time and space
        else:
            # Raise error if solution is 2D or 3D in space 
            if len(solution.shape) > 2:
                raise ValueError("Interpolation of solutions of 2D and 3D "+ 
                                 "space dimensions based on the provided "+
                                 "grid_obs and time_obs are not supported. "+
                                 "You can, instead, pass a custom "+
                                 "observation_map and pass grid_obs and "+
                                 "time_obs as None.")
            
            # Interpolate solution in space and time to the observation
            # time and space
            solution_obs = scipy.interpolate.RectBivariateSpline(
                self.grid_sol, self.time_steps, solution)(self.grid_obs,
                                                          self._time_obs)

        # Apply observation map
        if self.observation_map is not None:
            solution_obs = self.observation_map(solution_obs)
        
        # squeeze if only one time observation
        if len(self._time_obs) == 1:
            solution_obs = solution_obs.squeeze()

        return solution_obs
