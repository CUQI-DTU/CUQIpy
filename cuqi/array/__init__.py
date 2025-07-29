"""
CUQIpy Array Backend Module

This module provides a backend-agnostic interface for array operations.
It allows switching between different array backends like NumPy, PyTorch, CuPy, JAX
through environment variables or programmatic configuration.

Usage:
    import cuqi.array as xp
    
    # Use array operations
    x = xp.array([1, 2, 3])
    y = xp.zeros((3, 3))
    z = xp.dot(x, y)

Backend Selection:
    Set the CUQI_ARRAY_BACKEND environment variable or use set_backend():
    - "numpy" (default): Use NumPy backend
    - "pytorch"/"torch": Use PyTorch backend (GPU + gradients)
    - "cupy": Use CuPy backend (GPU acceleration)
    - "jax": Use JAX backend (JIT compilation)
    
    Example:
        export CUQI_ARRAY_BACKEND=pytorch
        # or
        xp.set_backend("pytorch")
"""

import os
from ._array import CUQIarray

# Backend selection mechanism
_BACKEND_NAME = os.getenv("CUQI_ARRAY_BACKEND", "numpy").lower()
_backend_module = None

def get_backend_name():
    """Get the current backend name."""
    return _BACKEND_NAME

def set_backend(backend_name):
    """Set the array backend programmatically."""
    global _BACKEND_NAME, _backend_module
    original_backend = _BACKEND_NAME
    _BACKEND_NAME = backend_name.lower()
    try:
        _backend_module = _load_backend()
        _expose_backend_functions()
    except ImportError:
        # Restore original backend name if loading fails
        _BACKEND_NAME = original_backend
        raise

def _load_backend():
    """Load the specified backend module."""
    if _BACKEND_NAME in ["numpy", "np"]:
        from ._numpy import load_backend
        return load_backend()
    elif _BACKEND_NAME in ["pytorch", "torch"]:
        from ._pytorch import load_backend
        return load_backend()
    elif _BACKEND_NAME in ["cupy", "cp"]:
        # For now, fall back to numpy for cupy
        from ._numpy import load_backend
        return load_backend()
    elif _BACKEND_NAME in ["jax", "jnp"]:
        # For now, fall back to numpy for jax
        from ._numpy import load_backend
        return load_backend()
    else:
        raise ValueError(f"Unsupported backend: {_BACKEND_NAME}")

def _expose_backend_functions():
    """Expose backend functions at module level."""
    if _BACKEND_NAME in ["numpy", "np"]:
        from ._numpy import get_backend_functions, to_numpy, pad
    elif _BACKEND_NAME in ["pytorch", "torch"]:
        from ._pytorch import get_backend_functions, to_numpy, pad
    elif _BACKEND_NAME in ["cupy", "cp"]:
        # For now, use numpy functions
        from ._numpy import get_backend_functions, to_numpy, pad
    elif _BACKEND_NAME in ["jax", "jnp"]:
        # For now, use numpy functions
        from ._numpy import get_backend_functions, to_numpy, pad
    else:
        raise ValueError(f"Unsupported backend: {_BACKEND_NAME}")
    
    # Get functions from backend
    functions = get_backend_functions(_backend_module)
    
    # Expose functions at module level
    globals().update(functions)
    
    # Add utility functions
    globals()['to_numpy'] = to_numpy
    globals()['pad'] = pad
    globals()['_backend_module'] = _backend_module
    
    # Add sparse module (delayed import to avoid circular imports)
    try:
        from . import _sparse
        globals()['sparse'] = _sparse
    except ImportError:
        # Handle case where sparse module might not be available
        pass

# Initialize backend on import
try:
    _backend_module = _load_backend()
    _expose_backend_functions()
except ImportError:
    # Fall back to NumPy if the default backend is not available
    if _BACKEND_NAME != "numpy":
        # Modify module-level variable
        globals()['_BACKEND_NAME'] = "numpy"
        _backend_module = _load_backend()
        _expose_backend_functions()
    else:
        raise

# Define what gets exported
__all__ = [
    # Backend control
    'get_backend_name', 'set_backend', 'to_numpy', 'pad',
    
    # Array creation
    'array', 'zeros', 'ones', 'zeros_like', 'ones_like', 'empty', 'empty_like',
    'full', 'full_like', 'arange', 'linspace', 'logspace', 'eye', 'identity',
    'diag', 'diagonal', 'meshgrid',
    
    # Shape manipulation
    'reshape', 'ravel', 'flatten', 'transpose', 'swapaxes', 'moveaxis',
    'flip', 'flipud', 'fliplr', 'rot90', 'roll',
    
    # Array joining and splitting
    'concatenate', 'stack', 'vstack', 'hstack', 'dstack',
    'split', 'hsplit', 'vsplit', 'dsplit',
    
    # Mathematical functions
    'sum', 'prod', 'mean', 'std', 'var', 'min', 'max', 'argmin', 'argmax',
    'sort', 'argsort', 'any', 'all', 'argwhere', 'cumsum', 'cumprod',
    'diff', 'gradient', 'maximum', 'minimum', 'repeat', 'isclose',
    'percentile', 'median', 'multiply', 'tile', 'float_power', 'piecewise',
    
    # Linear algebra
    'dot', 'matmul', 'inner', 'outer', 'cross', 'tensordot', 'einsum',
    'tril', 'triu', 'linalg',
    
    # Trigonometric functions
    'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan', 'arctan2',
    'sinh', 'cosh', 'tanh',
    
    # Exponential and logarithmic functions
    'exp', 'exp2', 'log', 'log2', 'log10', 'sqrt', 'square', 'power',
    'abs', 'sign',
    
    # Rounding functions
    'floor', 'ceil', 'round', 'clip',
    
    # Logic functions
    'where', 'isnan', 'isinf', 'isfinite', 'count_nonzero', 'allclose',
    'array_equiv', 'array_equal', 'isscalar', 'sinc', 'fix',
    
    # Complex numbers
    'real', 'imag', 'conj', 'angle', 'absolute',
    
    # Array conversion
    'asarray', 'asanyarray', 'ascontiguousarray', 'asfortranarray', 'copy',
    
         # Data types and constants
     'finfo', 'iinfo', 'newaxis', 'inf', 'nan', 'pi', 'e', 'size', 'shape',
     'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64',
     'float16', 'float32', 'float64', 'complex64', 'complex128', 'bool_',
     'ndarray', 'dtype', 'integer', 'floating', 'complexfloating',
    
         # Modules
     'random', 'fft', 'polynomial', 'sparse',
    
    # CUQIarray class
    'CUQIarray',
]
