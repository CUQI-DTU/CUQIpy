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
    pass

class CUQIarray2(ABC):
    pass
