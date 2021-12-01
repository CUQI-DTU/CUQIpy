import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse import hstack
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from cuqi.pde import PDE

import cuqi
from cuqi.samples import Samples
from cuqi.model import Model
import warnings

try: 
    import dolfin as dl
    import ufl
except Exception as error:
    warnings.warn(error.msg)

class FEniCSPDE(PDE):
    def __init__(self, PDE_form, mesh, solution_function_space, parameter_function_space, dirichlet_bc,observation_operator=None):
        self.PDE_form = PDE_form # function of PDE_solution, PDE_parameter, test_function
        self.mesh = mesh 
        self.solution_function_space  = solution_function_space
        self.parameter_function_space = parameter_function_space
        self.dirichlet_bc  = dirichlet_bc
        self.observation_operator = observation_operator

    def assemble(self,PDE_parameter):
        pass

    def solve(self):
        pass

    def observe(self,PDE_solution):
        pass


class SteadyStateLinearFEniCSPDE(FEniCSPDE):
    def __init__(self, PDE_form, mesh, solution_function_space, parameter_function_space, dirichlet_bc,observation_operator=None):
        super().__init__(PDE_form, mesh, solution_function_space, parameter_function_space, dirichlet_bc,observation_operator=observation_operator)

    def assemble(self,PDE_parameter_dof):
        PDE_parameter_fun = dl.Function(self.parameter_function_space)
        PDE_parameter_fun.vector().set_local(PDE_parameter_dof) 
        solution_trial_function = dl.TrialFunction(self.solution_function_space)
        solution_test_function = dl.TestFunction(self.solution_function_space)
        self.diff_op, self.rhs  = \
            dl.lhs(self.PDE_form(solution_trial_function,PDE_parameter_fun,solution_test_function)),\
            dl.rhs(self.PDE_form(solution_trial_function,PDE_parameter_fun,solution_test_function))
        self.PDE_parameter_fun = PDE_parameter_fun 

    def solve(self):
        PDE_solution_fun = dl.Function(self.solution_function_space)
        dl.solve(self.diff_op ==  self.rhs, PDE_solution_fun, self.dirichlet_bc)
        return PDE_solution_fun 

    def observe(self,PDE_solution_fun):
        if self.observation_operator == None: 
            return PDE_solution_fun.vector().get_local()
        else:
            return self._apply_obs_op(PDE_solution_fun,self.PDE_parameter_fun)

    def _apply_obs_op(self, PDE_parameter_fun, PDE_solution_fun,):
        obs = self.observation_operator(PDE_parameter_fun, PDE_solution_fun)
        if isinstance(obs, ufl.algebra.Operator):
            return dl.project(obs, self.solution_function_space).vector().get_local()
        elif isinstance(obs, dl.function.function.Function):
            return obs.vector().get_local()
        elif isinstance(obs, (np.ndarray, int, float)):
            return obs
        else:
            raise NotImplementedError("obs_op output must be a number, a numpy array or a ufl.algebra.Operator type")