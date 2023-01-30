from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import cuqi

__all__ = ["MultiVector", "Samples", "TimeSeries", "CUQIarray2"]

class MultiVector(ABC):
    pass

class Samples(MultiVector):
    pass

class TimeSeries(MultiVector):
    """
    Abstract base class for time series.

    times: Time points of the time series.
    values: Values of the time series.
    """
    pass

class TimeDependantPDESolution(TimeSeries):
    """
    Abstract base class for time series of PDEs.

    rejected_times: Time points of the time series that were rejected.
    local_errors: Estimated local errors of the time steps.
    u_list: List of solutions at the time points.
    t_list: List of time steps.
    """
    pass

class CUQIarray2(ABC):
    pass
