class Observer():
    pass

class SteadyObserver(observer):
    """
    Attributes:
    observation_operator: Observation operator (lambda function or x coordinates).
    observed_solution: Observed solution (CUQIarray)."""
    pass

class TimeDependantObserver(observer):
    """
    Attributes:
    observation_operator: Observation operator (lambda function or x and t coordinates).
    observed_solution: Observed solution (TimeSeries or CUQIArray)."""

    pass
