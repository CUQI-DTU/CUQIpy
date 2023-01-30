class Observer():
    pass

class SteadyObserver(Observer):
    """
    Attributes:
    observation_operator: Observation operator (lambda function or x coordinates).
    observed_solution: Observed solution (CUQIarray)."""
    pass

class TimeDependantObserver(Observer):
    """
    Attributes:
    observation_operator: Observation operator (lambda function or x and t coordinates).
    observed_solution: Observed solution (TimeSeries or CUQIArray)."""

    pass
