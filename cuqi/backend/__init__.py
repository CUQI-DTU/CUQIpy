# CUQIpy backend abstraction module

from ._backend import set, array, zeros, asarray, is_array, as_numpy

# Create a unified backend interface
class BackendInterface:
    """Unified backend interface that can be imported as xp"""
    
    def __init__(self):
        self._update_backend()
    
    def _update_backend(self):
        """Update the backend functions to use current backend"""
        from ._backend import array, zeros, asarray, is_array, as_numpy
        self.array = array
        self.zeros = zeros
        self.asarray = asarray
        self.is_array = is_array
        self.as_numpy = as_numpy

# Create the xp interface
xp = BackendInterface()

# Re-export the main functions
__all__ = ['set', 'array', 'zeros', 'asarray', 'is_array', 'as_numpy', 'xp'] 