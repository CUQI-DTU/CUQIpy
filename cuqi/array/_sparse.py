"""
Sparse matrix operations for CUQIpy array backends.

This module provides a unified interface for sparse matrix operations
across different backends, primarily using scipy.sparse as the backend.
"""


class SparseModule:
    """Sparse operations module for array backends."""
    
    def __init__(self, backend_functions):
        # Store backend-specific sparse functions
        self._backend_functions = backend_functions
        
    def __getattr__(self, name):
        """Get sparse operations from backend or scipy.sparse."""
        # First check if it's a backend-specific function
        if name in self._backend_functions:
            return self._backend_functions[name]
        
        # Otherwise, forward to scipy.sparse
        import scipy.sparse
        return getattr(scipy.sparse, name)
    
    @property
    def spdiags(self):
        """Create sparse diagonal matrix."""
        return self._backend_functions.get('sparse_spdiags')
    
    @property  
    def eye(self):
        """Create sparse identity matrix."""
        return self._backend_functions.get('sparse_eye')
        
    @property
    def kron(self):
        """Kronecker product of sparse matrices."""
        return self._backend_functions.get('sparse_kron')
        
    @property
    def vstack(self):
        """Stack sparse matrices vertically."""
        return self._backend_functions.get('sparse_vstack')


def create_sparse_module(backend_functions):
    """Create a sparse module for the given backend functions."""
    return SparseModule(backend_functions)