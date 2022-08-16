import numpy as np
from abc import ABC, abstractmethod
from ..pde import PDE
from ..samples import CUQIarray
import dolfin as dl
import ufl

__all__ = [
    'FEniCSPDE',
    'SteadyStateLinearFEniCSPDE'
]

class FEniCSPDE(PDE,ABC):
    def __init__(self, PDE_form, mesh, solution_function_space, parameter_function_space, dirichlet_bc,observation_operator=None):
        self.PDE_form = PDE_form # function of PDE_solution, PDE_parameter, test_function
        self.mesh = mesh 
        self.solution_function_space  = solution_function_space
        self.parameter_function_space = parameter_function_space
        self.dirichlet_bc  = dirichlet_bc
        self.observation_operator = self._create_observation_operator(observation_operator)

    @abstractmethod
    def assemble(self,parameter):
        pass

    @abstractmethod
    def solve(self):
        pass

    @abstractmethod
    def observe(self,PDE_solution):
        pass

    @abstractmethod
    def _create_observation_operator(self, observation_operator):
        pass

class SteadyStateLinearFEniCSPDE(FEniCSPDE):
    def __init__(self, PDE_form, mesh, solution_function_space, parameter_function_space, dirichlet_bc,observation_operator=None):
        super().__init__(PDE_form, mesh, solution_function_space, parameter_function_space, dirichlet_bc,observation_operator=observation_operator)

    def assemble(self,parameter):
        if isinstance(parameter, CUQIarray): 
            PDE_parameter_fun = parameter.funvals
        elif isinstance(parameter, dl.function.function.Function): 
            PDE_parameter_fun = parameter
        else:
            raise ValueError("parameter should be of type 'CUQIarray' or 'dl.function.function.Function'")

        solution_trial_function = dl.TrialFunction(self.solution_function_space)
        solution_test_function = dl.TestFunction(self.solution_function_space)
        self.diff_op, self.rhs  = \
            dl.lhs(self.PDE_form(PDE_parameter_fun, solution_trial_function,solution_test_function)),\
            dl.rhs(self.PDE_form(PDE_parameter_fun, solution_trial_function, solution_test_function))
        self.PDE_parameter_fun = PDE_parameter_fun 

    def solve(self):
        PDE_solution_fun = dl.Function(self.solution_function_space)
        dl.solve(self.diff_op ==  self.rhs, PDE_solution_fun, self.dirichlet_bc)
        return PDE_solution_fun, None 

    def observe(self,PDE_solution_fun):
        if self.observation_operator == None: 
            return PDE_solution_fun
        else:
            return self._apply_obs_op(self.PDE_parameter_fun, PDE_solution_fun)

    def _apply_obs_op(self, PDE_parameter_fun, PDE_solution_fun,):
        obs = self.observation_operator(PDE_parameter_fun, PDE_solution_fun)
        if isinstance(obs, ufl.algebra.Operator):
            return dl.project(obs, self.solution_function_space)
        elif isinstance(obs, dl.function.function.Function):
            return obs
        elif isinstance(obs, (np.ndarray, int, float)):
            return obs
        else:
            raise NotImplementedError("obs_op output must be a number, a numpy array or a ufl.algebra.Operator type")
    

    def _create_observation_operator(self, observation_operator):
        """
        """
        if observation_operator == 'potential':
            observation_operator = lambda m, u: u 
        elif observation_operator == 'gradu_squared':
            observation_operator = lambda m, u: dl.inner(dl.grad(u),dl.grad(u))
        elif observation_operator == 'power_density':
            observation_operator = lambda m, u: m*dl.inner(dl.grad(u),dl.grad(u))
        elif observation_operator == 'sigma_u':
            observation_operator = lambda m, u: m*u
        elif observation_operator == 'sigma_norm_gradu':
            observation_operator = lambda m, u: m*dl.sqrt(dl.inner(dl.grad(u),dl.grad(u)))
        elif observation_operator == None or callable(observation_operator):
            observation_operator = observation_operator
        else:
            raise NotImplementedError
        return observation_operator