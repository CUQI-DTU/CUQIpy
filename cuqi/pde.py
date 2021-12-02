from abc import ABC, abstractmethod
import scipy
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
        m, n = len(self.grid_obs), len(self.grid_sol)
        if (m == n):            
            equal_arrays = (self.grid_obs == self.grid_sol).all()
        else:
            equal_arrays = False
            
        if (equal_arrays == False):
            solution_obs = interp1d(self.grid_sol, solution, kind='quadratic')(self.grid_obs)
        else:
            solution_obs = solution
            
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
        comparison = (self.grid_obs == self.grid_sol)
        equal_arrays = comparison.all()
        if (equal_arrays == False):
            solution_obs = interp1d(self.grid_sol, solution, kind='quadratic')(self.grid_obs)
        else:
            solution_obs = solution
            
        return solution_obs