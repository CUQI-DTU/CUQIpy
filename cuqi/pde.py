from abc import ABC, abstractmethod
import scipy
from inspect import getsource
from scipy.interpolate import interp1d
import numpy as np

class PDE(ABC):
    """
    Parametrized PDE abstract base class
    """

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
        self._grids_equal = self._compare_grid(value,self.grid_sol)
        self._grid_obs = value

    @property
    def grids_equal(self):
        return self._grids_equal

class SteadyStateLinearPDE(PDE):
    """Linear steady state PDE.
    
    Parameters
    -----------   
    PDE_form : callable function
        Callable function which returns a tuple with
        the discretized differential operator A and right-hand-side b. The type of A and b are determined by what the method :meth:`linalg_solve` accepts as first and second parameters, respectively. 

    linalg_solve: lambda function or function handle
        linear system solver function with the signature :meth:`x, val1, val2, ...=linalg_solve(A,b,**linalg_solve_kwargs)` where A is the linear operator and b is the right hand side. `linalg_solve_kwargs` is any keywords arguments that the function :meth:`linalg_solve` can take. x is the solution of A*x=b of type `numpy.ndarray`. val1, val2, etc. are optional and can be a one or more values the solver return, e.g. information and number of iterations (for iterative solvers). If linalg_solve is None, :meth:`scipy.linalg.solve` will be used. 

    linalg_solve_kwargs: a dictionary 
        A dictionary of the keywords arguments that linalg_solve can take.

    observation_map: a function handle
        A function that takes the PDE solution as input and the returns the observed solution. e.g. `observation_map=lambda u: u**2` or `observation_map=lambda u: u[0]`

    grid_sol: np.ndarray
        The grid on which solution is defined

    grid_obs: np.ndarray
        The grid on which the observed solution should be interpolated (Currently only supported for 1D problems).  

    Example
    -------- 
    See demo demos/demo24_fwd_poisson.py for an illustration on how to use SteadyStateLinearPDE with varying solver choices. And demos demos/demo25_fwd_poisson_2D.py and demos/demo26_fwd_poisson_mixedBC.py for examples with mixed (Dirichlet and Neumann) boundary conditions problems. demos/demo25_fwd_poisson_2D.py also illustrates how to observe on a specific boundary, for example.
    """

    def __init__(self, PDE_form, grid_sol=None, grid_obs=None, observation_map=None, linalg_solve=None, linalg_solve_kwargs={}):
        self.PDE_form = PDE_form
        self.grid_sol = grid_sol
        self.grid_obs = grid_obs
        if linalg_solve == None:
            linalg_solve = scipy.linalg.solve
        self._linalg_solve = linalg_solve
        self._linalg_solve_kwargs = linalg_solve_kwargs
        self.observation_map = observation_map 

    def assemble(self, parameter):
        """Assembles differential operator and rhs according to PDE_form"""
        self.diff_op, self.rhs = self.PDE_form(parameter)

    def solve(self):
        """Solve the PDE and returns the solution and an information variable `info` which is a tuple of all variables returned by the function `linalg_solve` after the solution."""
        if not hasattr(self,"diff_op") or not hasattr(self,"rhs"):
            raise Exception("PDE is not assembled.")

        returned_values = self._linalg_solve(self.diff_op, self.rhs, **self._linalg_solve_kwargs)
        if isinstance(returned_values, tuple):
            solution = returned_values[0]
            info = returned_values[1:]
        else:
            solution = returned_values
            info = None

        return solution, info

    def observe(self, solution):
            
        if self.grids_equal:
            solution_obs = solution
        else:
            solution_obs = interp1d(self.grid_sol, solution, kind='quadratic')(self.grid_obs)

        if self.observation_map is not None:
            solution_obs = self.observation_map(solution_obs)
                
        return solution_obs
        
class TimeDependentLinearPDE(PDE):
    """Time steady state PDE.
    
    Parameters
    -----------   
    PDE_form : callable function
        Callable function which returns a tuple with 3 things:
        1: matrix used for the time-stepping.
        2: initial condition
        3: time_steps array

    Example
    -----------  
    <<< ....
    <<< ....
    <<< ....
    """
    def __init__(self, PDE_form, grid_sol=None, grid_obs=None, observation_map=None):
        self.PDE_form = PDE_form
        self.grid_sol = grid_sol
        self.grid_obs = grid_obs
        self.observation_map = observation_map

    def assemble(self, parameter):
        """Assemble PDE"""
        self.diff_op, self.IC, self.time_steps = self.PDE_form(parameter)

    def solve(self):
        """Solve PDE by time-stepping"""
        u = self.IC
        for t in self.time_steps:
            u = self.diff_op@u
        
        info = None
        return u, info

    def observe(self, solution):
            
        if self.grids_equal:
            solution_obs = solution
        else:
            solution_obs = interp1d(self.grid_sol, solution, kind='quadratic')(self.grid_obs)

        if self.observation_map is not None:
            solution_obs = self.observation_map(solution_obs)
            
        return solution_obs