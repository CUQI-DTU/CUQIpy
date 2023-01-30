# do not import numpy
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from typing import Optional
import cuqi

class TimeIntegrator(ABC):
    @abstractmethod
    def propagate(self, u, dt, rhs, diff_op, solve_linear_system, I=None, apply_bc=None):
        """Propagate the solution using the explicit Euler method."""
        pass

class ThetaMethod(TimeIntegrator):
    def __init__(self, theta):
	    # solver details
        self.theta = theta # 0: backward Euler, 1: forward Euler, 0.5: trapezoidal  

    def propagate(self, u, dt, rhs, diff_op, solver, I=None, apply_bc=None):
        if I is None:
            I = np.eye(len(u))
        rhs_op = I + self.theta*dt*diff_op
        lhs_op = I - (1-self.theta)*dt*diff_op
        u_next, info = solver(rhs_op, lhs_op@u + dt*rhs)
        return u_next, info

class ForwardEuler(ThetaMethod):
    def __init__(self):
        super().__init__(1.0)

class BackwardEuler(ThetaMethod):
    def __init__(self):
        super().__init__(0.0)

class Trapezoidal(ThetaMethod):
    def __init__(self):
        super().__init__(0.5)
