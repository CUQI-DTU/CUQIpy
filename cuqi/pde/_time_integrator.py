# do not import numpy
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from typing import Optional
import cuqi

class TimeIntegrator(ABC):
    """
    Abstract base class for time integrators.
    Attributes:
    initial_condition: Initial condition for the time integrator.
    initial_time_step: Initial time step size.
    time_step_selector: Time_step selector object (default, static).
    solution: Time series of the solution, by default saves a window of
    solutions required by the time integrator, can be sat to save the
    entire solution. Information about what to save are contained within.
    observer: Observer object that contains the information about the
    observation operator and the stores the observed solution.


    Methods:
    __init__
    step
    assemble_step
    propagate_step (solve)
    observe_step
    approximate_local_error

    Design Notes:
    - The time integrator can be called TimeStep
    """
    pass
#    def __init__(self, initial_condition, initial_time_step, time_step_selector, observer):
#        self._initial_condition = initial_condition
#        self._initial_time_step = initial_time_step
#        self.time_step_selector = time_step_selector
#        self._solution = None # create solution object
#        self.observer = observer
#        self.method = 'forward_euler' # this option to be removed
#
#        self._current_time = None
#	
#
#
#
#
#    def step(self):
#        """Take one time step."""
#        self._assemble_step()
#        self._propagate_step()
#        self._observe_step()
#        self._approximate_local_error()
#
#    def _propagate_step(self):
#        """From current time solution, compute the next time solution."""
#        if self.method == 'forward_euler':
#            for idx, t in enumerate(self.time_steps[:-1]):
#                dt = self.time_steps[idx+1] - t
#                self.assemble_step(t)
#                if idx == 0:
#                    u = self.initial_condition
#                u = (dt*self.diff_op + np.eye(len(u)))@u + dt*self.rhs  # from u at time t, gives u at t+dt
#            info = None
#
#        if self.method == 'backward_euler':
#            for idx, t in enumerate(self.time_steps[1:]):
#                dt = t - self.time_steps[idx]
#                self.assemble_step(t)
#                if idx == 0:
#                    u = self.initial_condition
#                A = np.eye(len(u)) - dt*self.diff_op
#                # from u at time t-dt, gives u at t
#                u, info = self._solve_linear_system(
#                    A, u + dt*self.rhs, self._linalg_solve, self._linalg_solve_kwargs)
#
#        return u, info

#Auto generated method:
#@staticmethod
#def create_time_step_selector(time_step_selector, initial_time_step, time_steps):
#    """Create a time step selector object."""
#    if time_step_selector is None:
#	time_step_selector = StaticTimeStepSelector(initial_time_step)
#    elif isinstance(time_step_selector, str):
#	if time_step_selector == 'static':
#	    time_step_selector = StaticTimeStepSelector(initial_time_step)
#	elif time_step_selector == 'adaptive':
#	    time_step_selector = AdaptiveTimeStepSelector(time_steps)
#	else:
#	    raise ValueError('Unknown time step selector.')
#    elif not isinstance(time_step_selector, TimeStepSelector):
#	raise TypeError('Unknown time step selector.')
#    return time_step_selector


class AlphaMethod(TimeIntegrator):
    def __init__(self, alpha = 0):
        self.alpha = alpha
        pass


class BackwardEuler(AlphaMethod):
    def __init__(self):
        pass
    #TODO: update signature to remove solve_linear_system
    def propagate(self, u: cuqi.samples.CUQIarray, dt, rhs: cuqi.samples.CUQIarray, diff_op, solve_linear_system):
        """Propagate the solution using the explicit Euler method.
	
	Design Notes:
	u can be sol obj if multiple time steps are required to compute the next time step."""
        I = np.eye(len(u))
        A = I - dt*diff_op
        u_next, info = solve_linear_system(
                A, u + dt*rhs)
        return u_next, info

class ForwardEuler(AlphaMethod):
    def __init__(self):
        pass

    def propagate(self, u: cuqi.samples.CUQIarray, dt, rhs: cuqi.samples.CUQIarray, diff_op, solve_linear_system=None):
        """Propagate the solution using the explicit Euler method.
	
	Design Notes:
	u can be sol obj if multiple time steps are required to compute the next time step."""
        I = np.eye(len(u))
        u_next = (dt*diff_op + I)@u + dt*rhs
        return u_next, None

class StormerVerlet(TimeIntegrator):
    pass


def get_integrator_object(value):
    if value.lower() == 'forward_euler':
        integrator = ForwardEuler()
    elif value.lower() == 'backward_euler':
        integrator = BackwardEuler()
    else:
        raise ValueError(
            "method can be set to either `forward_euler` or `backward_euler`")
    return integrator
