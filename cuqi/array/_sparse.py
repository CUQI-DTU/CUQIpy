"""
Sparse matrix operations for CUQIpy array backends.

This module provides a unified interface for sparse matrix operations
across different backends, primarily using scipy.sparse as the backend.
"""


class BackendSparseMatrix:
    """Backend-agnostic wrapper for sparse matrices that provides consistent interface across NumPy and PyTorch backends."""
    
    def __init__(self, scipy_sparse_matrix):
        """Initialize with a scipy sparse matrix."""
        self._scipy_matrix = scipy_sparse_matrix
        
    @property
    def T(self):
        """Transpose of the sparse matrix."""
        return BackendSparseMatrix(self._scipy_matrix.T)
    
    @property
    def shape(self):
        """Shape of the sparse matrix."""
        return self._scipy_matrix.shape
    
    @property
    def nnz(self):
        """Number of non-zero elements in the sparse matrix."""
        return self._scipy_matrix.nnz
    
    def diagonal(self):
        """Get diagonal elements of the sparse matrix."""
        # Import here to avoid circular imports
        import cuqi.array as xp
        diag_elements = self._scipy_matrix.diagonal()
        # Convert to current backend
        return xp.array(diag_elements)
    
    def __mul__(self, other):
        """Multiplication with scalars or arrays."""
        # Import here to avoid circular imports
        import cuqi.array as xp
        
        if xp.isscalar(other):
            # Scalar multiplication returns another BackendSparseMatrix
            result = other * self._scipy_matrix
            return BackendSparseMatrix(result)
        else:
            # Array multiplication - return the result as an array
            if xp.get_backend_name() == "pytorch":
                # Convert other to numpy for scipy operations
                if hasattr(other, 'cpu'):  # PyTorch tensor
                    other_np = xp.to_numpy(other)
                else:
                    other_np = other
                result = self._scipy_matrix @ other_np
                # Convert back to current backend
                return xp.array(result)
            else:
                # NumPy backend
                result = self._scipy_matrix @ other
                return result
    
    def __rmul__(self, other):
        """Right multiplication with scalars or arrays."""
        # Import here to avoid circular imports
        import cuqi.array as xp
        
        # Check if other is a scalar (including 0-dimensional arrays)
        if xp.isscalar(other) or (hasattr(other, 'ndim') and other.ndim == 0):
            # Scalar multiplication
            if hasattr(other, 'item'):  # Extract scalar from 0-dim array/tensor
                other = other.item()
            result = other * self._scipy_matrix
            return BackendSparseMatrix(result)
        else:
            # Array multiplication from the left
            if xp.get_backend_name() == "pytorch":
                # Convert other to numpy for scipy operations
                if hasattr(other, 'cpu'):  # PyTorch tensor
                    other_np = xp.to_numpy(other)
                else:
                    other_np = other
                result = other_np @ self._scipy_matrix
                # Convert back to current backend
                return xp.array(result)
            else:
                # NumPy backend
                result = other @ self._scipy_matrix
                return result
    
    def __matmul__(self, other):
        """Matrix multiplication using @ operator."""
        # Import here to avoid circular imports
        import cuqi.array as xp
        
        if xp.get_backend_name() == "pytorch":
            # Convert other to numpy for scipy operations
            if hasattr(other, 'cpu'):  # PyTorch tensor
                other_np = xp.to_numpy(other)
            else:
                other_np = other
            result = self._scipy_matrix @ other_np
            # Convert back to current backend
            return xp.array(result)
        else:
            # NumPy backend
            result = self._scipy_matrix @ other
            return result
    
    def __rmatmul__(self, other):
        """Right matrix multiplication using @ operator."""
        # Import here to avoid circular imports
        import cuqi.array as xp
        
        if xp.get_backend_name() == "pytorch":
            # Convert other to numpy for scipy operations
            if hasattr(other, 'cpu'):  # PyTorch tensor
                other_np = xp.to_numpy(other)
            else:
                other_np = other
            result = other_np @ self._scipy_matrix
            # Convert back to current backend
            return xp.array(result)
        else:
            # NumPy backend
            result = other @ self._scipy_matrix
            return result
    
    def toarray(self):
        """Convert sparse matrix to dense array."""
        return self._scipy_matrix.toarray()
    
    def get_scipy_matrix(self):
        """Get the underlying scipy sparse matrix for operations that require it."""
        return self._scipy_matrix


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
    
    # Expose BackendSparseMatrix
    BackendSparseMatrix = BackendSparseMatrix


def create_sparse_module(backend_functions):
    """Create a sparse module for the given backend functions."""
    return SparseModule(backend_functions)