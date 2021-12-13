from abc import ABC, abstractmethod
import scipy
from inspect import getsource
from scipy.interpolate import interp1d

class PDE(ABC):
    """Parametrized PDE abstract base class"""

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
        the discretized differential operator and right-hand-side.

    Example
    -----------  
    <<< ....
    <<< ....
    <<< ....
    """
    def __init__(self, PDE_form, grid_sol=None, grid_obs=None):
        self.PDE_form = PDE_form
        self.grid_sol = grid_sol
        self.grid_obs = grid_obs

    def assemble(self, parameter):
        """Assembles differential operator and rhs according to PDE_form"""
        self.diff_op, self.rhs = self.PDE_form(parameter)

    def solve(self):
        """Solve PDE and return solution"""
        if not hasattr(self,"diff_op") or not hasattr(self,"rhs"):
            raise Exception("PDE is not assembled.")

        solution = scipy.linalg.solve(self.diff_op,self.rhs)

        return solution

    def observe(self, solution):
            
        if self.grids_equal:
            solution_obs = solution
        else:
            solution_obs = interp1d(self.grid_sol, solution, kind='quadratic')(self.grid_obs)
            
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
    def __init__(self, PDE_form, grid_sol=None, grid_obs=None):
        self.PDE_form = PDE_form
        self.grid_sol = grid_sol
        self.grid_obs = grid_obs

    def assemble(self, parameter):
        """Assemble PDE"""
        self.diff_op, self.IC, self.time_steps = self.PDE_form(parameter)

    def solve(self):
        """Solve PDE by time-stepping"""
        u = self.IC
        for t in self.time_steps:
            u = self.diff_op@u
        return u

    def observe(self, solution):
            
        if self.grids_equal:
            solution_obs = solution
        else:
            solution_obs = interp1d(self.grid_sol, solution, kind='quadratic')(self.grid_obs)
            
        return solution_obs