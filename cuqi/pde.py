from abc import ABC, abstractmethod
import scipy

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


class LinearSteadyStatePDE(PDE):
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
    def __init__(self,PDE_form):
        self.PDE_form = PDE_form

    def assemble(self, parameter):
        """Assembles differential operator and rhs according to PDE_form"""
        self.diff_op, self.rhs = self.PDE_form(parameter)

    def solve(self):
        """Solve PDE and return solution"""
        if not hasattr(self,"diff_op") or not hasattr(self,"rhs"):
            raise Exception("PDE is not assembled.")

        solution = scipy.linalg.solve(self.diff_op,self.rhs)

        return solution

    def observe(self,solution):
        return solution
        
class TimeDependentLinearPDE(PDE):
    """Time steady state PDE.
    
    Parameters
    -----------   
    PDE_form : callable function
        Callable function which returns a matrix used for the time-stepping.

    time_steps : array, list
        Array with the time steps.

    Example
    -----------  
    <<< ....
    <<< ....
    <<< ....
    """
    def __init__(self,PDE_form,time_steps):
        self.diff_op = PDE_form()
        self.time_steps = time_steps

    def assemble(self, parameter):
        """Sets initial condition"""
        self.u0 = parameter #Set initial condition

    def solve(self):
        """Solve PDE by time-stepping"""
        u = self.u0
        for t in self.time_steps:
            u = self.diff_op@u
        return u

    def observe(self,solution):
        return solution