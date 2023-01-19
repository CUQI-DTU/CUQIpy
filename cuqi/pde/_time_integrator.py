# do not import numpy


from abc import ABC, abstractmethod

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
    solve_step
    observe_step
    approximate_local_error
    """
    pass

class AlphaMethod(TimeIntegrator):
    pass

class ImplicitEuler(AlphaMethod):
    pass

class ExplicitEuler(AlphaMethod):
    pass

class StormerVerlet(TimeIntegrator):
    pass


