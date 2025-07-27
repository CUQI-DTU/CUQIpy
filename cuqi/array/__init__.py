"""
CUQIpy Array Backend Module

This module provides a backend-agnostic interface for array operations.
It allows switching between different array backends like NumPy, CuPy, JAX, etc.
through environment variables or configuration.

Usage:
    import cuqi.array as xp
    
    # Use array operations
    x = xp.array([1, 2, 3])
    y = xp.zeros((3, 3))
    z = xp.dot(x, y)

Backend Selection:
    Set the CUQI_ARRAY_BACKEND environment variable:
    - "numpy" (default): Use NumPy backend
    - "cupy": Use CuPy backend (GPU acceleration)
    - "jax": Use JAX backend (JIT compilation)
    
    Example:
        export CUQI_ARRAY_BACKEND=cupy
"""

import os
import warnings
from ._array import CUQIarray

# Backend selection mechanism
_BACKEND_NAME = os.getenv("CUQI_ARRAY_BACKEND", "numpy").lower()

def get_backend_name():
    """Get the current backend name."""
    return _BACKEND_NAME

def set_backend(backend_name):
    """Set the array backend programmatically.
    
    Parameters
    ----------
    backend_name : str
        Name of the backend to use ("numpy", "cupy", "jax")
    """
    global _BACKEND_NAME, _backend_module
    _BACKEND_NAME = backend_name.lower()
    _backend_module = _load_backend(_BACKEND_NAME)
    _expose_backend_functions()

def _load_backend(backend_name):
    """Load the specified backend module."""
    if backend_name == "numpy":
        import numpy as backend_module
        return backend_module
    elif backend_name == "cupy":
        try:
            import cupy as backend_module
            return backend_module
        except ImportError:
            warnings.warn("CuPy not available, falling back to NumPy")
            import numpy as backend_module
            return backend_module
    elif backend_name == "jax":
        try:
            import jax.numpy as backend_module
            return backend_module
        except ImportError:
            warnings.warn("JAX not available, falling back to NumPy")
            import numpy as backend_module
            return backend_module
    elif backend_name == "pytorch" or backend_name == "torch":
        try:
            import torch as backend_module
            return backend_module
        except ImportError:
            warnings.warn("PyTorch not available, falling back to NumPy")
            import numpy as backend_module
            return backend_module
    else:
        raise ValueError(f"Unsupported array backend '{backend_name}'. "
                        f"Supported backends: numpy, cupy, jax, pytorch")

# Load the initial backend
_backend_module = _load_backend(_BACKEND_NAME)

def _expose_backend_functions():
    """Expose backend functions at module level."""
    global array, zeros, ones, zeros_like, ones_like, empty, empty_like, full, full_like
    global arange, linspace, logspace, eye, identity, diag, diagonal
    global reshape, ravel, flatten, transpose, swapaxes, moveaxis, shape, size
    global concatenate, stack, vstack, hstack, dstack, split, hsplit, vsplit, dsplit
    global sum, prod, mean, std, var, min, max, argmin, argmax, sort, argsort
    global dot, matmul, inner, outer, cross, tensordot, einsum
    global sin, cos, tan, arcsin, arccos, arctan, arctan2, sinh, cosh, tanh
    global exp, exp2, log, log2, log10, sqrt, square, power, abs, sign
    global floor, ceil, round, clip, where, isnan, isinf, isfinite, count_nonzero, allclose, array_equiv, array_equal
    global real, imag, conj, angle, absolute
    global random, linalg, fft
    global ndarray, dtype, newaxis, inf, nan, pi, e
    global asarray, asanyarray, ascontiguousarray, asfortranarray
    global copy, deepcopy, finfo, iinfo
    global int8, int16, int32, int64, uint8, uint16, uint32, uint64
    global float16, float32, float64, complex64, complex128
    global integer, floating, complexfloating
    
    # Core array creation and manipulation
    if _BACKEND_NAME == "pytorch" or _BACKEND_NAME == "torch":
        # PyTorch uses tensor instead of array
        array = lambda x, dtype=None: _backend_module.tensor(x, dtype=dtype)
        zeros = _backend_module.zeros
        ones = _backend_module.ones
        zeros_like = _backend_module.zeros_like
        ones_like = _backend_module.ones_like
        empty = _backend_module.empty
        empty_like = _backend_module.empty_like
        full = _backend_module.full
        full_like = _backend_module.full_like
    else:
        array = _backend_module.array
        zeros = _backend_module.zeros
        ones = _backend_module.ones
        zeros_like = _backend_module.zeros_like
        ones_like = _backend_module.ones_like
        empty = _backend_module.empty
        empty_like = _backend_module.empty_like
        full = _backend_module.full
        full_like = _backend_module.full_like
    
    # Array generation
    arange = _backend_module.arange
    if _BACKEND_NAME == "pytorch" or _BACKEND_NAME == "torch":
        linspace = lambda start, end, steps=50, **kwargs: _backend_module.linspace(start, end, steps, **kwargs)
    else:
        linspace = _backend_module.linspace
    eye = _backend_module.eye
    diag = _backend_module.diag
    if hasattr(_backend_module, 'diagonal'):
        diagonal = _backend_module.diagonal
    else:
        diagonal = lambda a, offset=0, axis1=0, axis2=1: _backend_module.diag(a, diagonal=offset)
    
    # Shape manipulation
    if _BACKEND_NAME == "pytorch" or _BACKEND_NAME == "torch":
        reshape = lambda x, *args: x.reshape(*args) if hasattr(x, 'reshape') else _backend_module.reshape(x, *args)
        ravel = lambda x: x.flatten() if hasattr(x, 'flatten') else _backend_module.flatten(x)
        flatten = lambda x: x.flatten() if hasattr(x, 'flatten') else _backend_module.flatten(x)
        transpose = lambda x, axes=None: x.permute(*reversed(range(x.ndim))) if axes is None else x.permute(*axes)
    else:
        reshape = _backend_module.reshape
        ravel = _backend_module.ravel
        flatten = _backend_module.ravel  # Some backends don't have flatten
        transpose = _backend_module.transpose
    
    # Array joining and splitting
    concatenate = _backend_module.concatenate
    stack = _backend_module.stack
    split = _backend_module.split
    
    # Mathematical operations
    if _BACKEND_NAME == "pytorch" or _BACKEND_NAME == "torch":
        # PyTorch has slightly different APIs
        sum = lambda x, axis=None, keepdims=False: _backend_module.sum(x, dim=axis, keepdim=keepdims) if axis is not None else _backend_module.sum(x)
        mean = lambda x, axis=None, keepdims=False: _backend_module.mean(x, dim=axis, keepdim=keepdims) if axis is not None else _backend_module.mean(x)
        std = lambda x, axis=None, keepdims=False: _backend_module.std(x, dim=axis, keepdim=keepdims) if axis is not None else _backend_module.std(x)
        var = lambda x, axis=None, keepdims=False: _backend_module.var(x, dim=axis, keepdim=keepdims) if axis is not None else _backend_module.var(x)
        min = lambda x, axis=None, keepdims=False: _backend_module.min(x, dim=axis, keepdim=keepdims)[0] if axis is not None else _backend_module.min(x)
        max = lambda x, axis=None, keepdims=False: _backend_module.max(x, dim=axis, keepdim=keepdims)[0] if axis is not None else _backend_module.max(x)
        argmin = lambda x, axis=None, keepdims=False: _backend_module.argmin(x, dim=axis, keepdim=keepdims)
        argmax = lambda x, axis=None, keepdims=False: _backend_module.argmax(x, dim=axis, keepdim=keepdims)
        sort = lambda x, axis=-1: _backend_module.sort(x, dim=axis)[0]
        argsort = lambda x, axis=-1: _backend_module.argsort(x, dim=axis)
    else:
        sum = _backend_module.sum
        mean = _backend_module.mean
        std = _backend_module.std
        var = _backend_module.var
        min = _backend_module.min
        max = _backend_module.max
        argmin = _backend_module.argmin
        argmax = _backend_module.argmax
        sort = _backend_module.sort
        argsort = _backend_module.argsort
    
    # Linear algebra
    if _BACKEND_NAME == "pytorch" or _BACKEND_NAME == "torch":
        # PyTorch uses different function names
        dot = lambda a, b: _backend_module.dot(a, b) if a.dim() == 1 and b.dim() == 1 else _backend_module.matmul(a, b)
    else:
        dot = _backend_module.dot
    
    # Mathematical functions
    if _BACKEND_NAME == "pytorch" or _BACKEND_NAME == "torch":
        # PyTorch functions need tensor inputs
        sin = lambda x: _backend_module.sin(_backend_module.tensor(x) if not isinstance(x, _backend_module.Tensor) else x)
        cos = lambda x: _backend_module.cos(_backend_module.tensor(x) if not isinstance(x, _backend_module.Tensor) else x)
        tan = lambda x: _backend_module.tan(_backend_module.tensor(x) if not isinstance(x, _backend_module.Tensor) else x)
        exp = lambda x: _backend_module.exp(_backend_module.tensor(x) if not isinstance(x, _backend_module.Tensor) else x)
        log = lambda x: _backend_module.log(_backend_module.tensor(x) if not isinstance(x, _backend_module.Tensor) else x)
        sqrt = lambda x: _backend_module.sqrt(_backend_module.tensor(x) if not isinstance(x, _backend_module.Tensor) else x)
        square = lambda x: _backend_module.square(_backend_module.tensor(x) if not isinstance(x, _backend_module.Tensor) else x)
        abs = lambda x: _backend_module.abs(_backend_module.tensor(x) if not isinstance(x, _backend_module.Tensor) else x)
        sign = lambda x: _backend_module.sign(_backend_module.tensor(x) if not isinstance(x, _backend_module.Tensor) else x)
        floor = lambda x: _backend_module.floor(_backend_module.tensor(x) if not isinstance(x, _backend_module.Tensor) else x)
        ceil = lambda x: _backend_module.ceil(_backend_module.tensor(x) if not isinstance(x, _backend_module.Tensor) else x)
        clip = lambda x, min_val, max_val: _backend_module.clamp(_backend_module.tensor(x) if not isinstance(x, _backend_module.Tensor) else x, min_val, max_val)
        where = lambda condition, x, y: _backend_module.where(condition, x, y)
    else:
        sin = _backend_module.sin
        cos = _backend_module.cos
        tan = _backend_module.tan
        exp = _backend_module.exp
        log = _backend_module.log
        sqrt = _backend_module.sqrt
        square = _backend_module.square
        abs = _backend_module.abs
        sign = _backend_module.sign
        floor = _backend_module.floor
        ceil = _backend_module.ceil
        clip = _backend_module.clip
        where = _backend_module.where
    
    # Type checking
    isnan = _backend_module.isnan
    isinf = _backend_module.isinf
    isfinite = _backend_module.isfinite
    
    # Counting and comparison functions
    if _BACKEND_NAME == "pytorch" or _BACKEND_NAME == "torch":
        count_nonzero = lambda x: _backend_module.count_nonzero(_backend_module.tensor(x) if not isinstance(x, _backend_module.Tensor) else x)
        allclose = lambda a, b, rtol=1e-05, atol=1e-08: _backend_module.allclose(
            _backend_module.tensor(a) if not isinstance(a, _backend_module.Tensor) else a,
            _backend_module.tensor(b) if not isinstance(b, _backend_module.Tensor) else b,
            rtol=rtol, atol=atol
        )
        array_equiv = lambda a, b: (lambda result: result.all() if hasattr(result, 'all') else result)(
            _backend_module.equal(
                _backend_module.tensor(a) if not isinstance(a, _backend_module.Tensor) else a,
                _backend_module.tensor(b) if not isinstance(b, _backend_module.Tensor) else b
            )
        )
        array_equal = lambda a, b: (lambda result: result.all() if hasattr(result, 'all') else result)(
            _backend_module.equal(
                _backend_module.tensor(a) if not isinstance(a, _backend_module.Tensor) else a,
                _backend_module.tensor(b) if not isinstance(b, _backend_module.Tensor) else b
            )
        )
    else:
        count_nonzero = _backend_module.count_nonzero if hasattr(_backend_module, 'count_nonzero') else lambda x: (x != 0).sum()
        allclose = _backend_module.allclose if hasattr(_backend_module, 'allclose') else lambda a, b, rtol=1e-05, atol=1e-08: (abs(a - b) <= atol + rtol * abs(b)).all()
        array_equiv = _backend_module.array_equiv if hasattr(_backend_module, 'array_equiv') else lambda a, b: (a == b).all()
        array_equal = _backend_module.array_equal if hasattr(_backend_module, 'array_equal') else lambda a, b: (a == b).all()
    
    # Complex number operations
    real = _backend_module.real
    imag = _backend_module.imag
    conj = _backend_module.conj
    absolute = _backend_module.absolute
    
    # Constants
    if _BACKEND_NAME == "pytorch" or _BACKEND_NAME == "torch":
        import math
        inf = float('inf')
        nan = float('nan')
        pi = math.pi
        e = math.e
        newaxis = None  # PyTorch uses None for new axis
        ndarray = _backend_module.Tensor  # PyTorch uses Tensor class
    else:
        inf = _backend_module.inf
        nan = _backend_module.nan
        pi = _backend_module.pi
        e = _backend_module.e
        newaxis = _backend_module.newaxis
        ndarray = _backend_module.ndarray
    
    # Type conversion
    if _BACKEND_NAME == "pytorch" or _BACKEND_NAME == "torch":
        asarray = lambda x, dtype=None: _backend_module.tensor(x, dtype=dtype) if not isinstance(x, _backend_module.Tensor) else x
        asanyarray = lambda x, dtype=None: _backend_module.tensor(x, dtype=dtype) if not isinstance(x, _backend_module.Tensor) else x
    else:
        asarray = _backend_module.asarray
        asanyarray = _backend_module.asanyarray
    
    # Handle backend-specific differences
    if hasattr(_backend_module, 'logspace'):
        logspace = _backend_module.logspace
    else:
        logspace = lambda start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0: \
            _backend_module.power(base, _backend_module.linspace(start, stop, num, endpoint, dtype, axis))
    
    if hasattr(_backend_module, 'identity'):
        identity = _backend_module.identity
    else:
        identity = lambda n, dtype=None: _backend_module.eye(n, dtype=dtype)
    
    if hasattr(_backend_module, 'vstack'):
        vstack = _backend_module.vstack
        hstack = _backend_module.hstack
    else:
        vstack = lambda tup: _backend_module.concatenate(tup, axis=0)
        hstack = lambda tup: _backend_module.concatenate(tup, axis=1)
    
    if hasattr(_backend_module, 'hsplit'):
        hsplit = _backend_module.hsplit
        vsplit = _backend_module.vsplit
    else:
        hsplit = lambda ary, indices_or_sections: _backend_module.split(ary, indices_or_sections, axis=1)
        vsplit = lambda ary, indices_or_sections: _backend_module.split(ary, indices_or_sections, axis=0)
    
    if hasattr(_backend_module, 'dstack'):
        dstack = _backend_module.dstack
        dsplit = _backend_module.dsplit
    else:
        dstack = lambda tup: _backend_module.concatenate([_backend_module.expand_dims(arr, axis=2) for arr in tup], axis=2)
        dsplit = lambda ary, indices_or_sections: _backend_module.split(ary, indices_or_sections, axis=2)
    
    if hasattr(_backend_module, 'swapaxes'):
        swapaxes = _backend_module.swapaxes
    else:
        swapaxes = lambda a, axis1, axis2: _backend_module.transpose(a, tuple(
            axis2 if i == axis1 else axis1 if i == axis2 else i 
            for i in range(a.ndim)
        ))
    
    if hasattr(_backend_module, 'moveaxis'):
        moveaxis = _backend_module.moveaxis
    else:
        def moveaxis(a, source, destination):
            # Simplified moveaxis implementation
            axes = list(range(a.ndim))
            axes.pop(source)
            axes.insert(destination, source)
            return _backend_module.transpose(a, axes)
    
    if hasattr(_backend_module, 'prod'):
        prod = _backend_module.prod
    else:
        prod = _backend_module.product
    
    if hasattr(_backend_module, 'matmul'):
        matmul = _backend_module.matmul
    else:
        matmul = _backend_module.dot
    
    if hasattr(_backend_module, 'inner'):
        inner = _backend_module.inner
    else:
        inner = lambda a, b: _backend_module.dot(a.ravel(), b.ravel())
    
    if hasattr(_backend_module, 'outer'):
        outer = _backend_module.outer
    else:
        outer = lambda a, b: _backend_module.multiply.outer(a.ravel(), b.ravel())
    
    if hasattr(_backend_module, 'cross'):
        cross = _backend_module.cross
    else:
        # Simplified cross product for 3D vectors
        def cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
            return _backend_module.array([
                a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
                a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
                a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
            ]).T
    
    if hasattr(_backend_module, 'tensordot'):
        tensordot = _backend_module.tensordot
    else:
        # Simplified tensordot
        def tensordot(a, b, axes=2):
            if axes == 2:
                return _backend_module.dot(a, b)
            else:
                # More complex cases would need full implementation
                return _backend_module.dot(a, b)
    
    if hasattr(_backend_module, 'einsum'):
        einsum = _backend_module.einsum
    else:
        # Fallback to basic operations for simple cases
        def einsum(subscripts, *operands, **kwargs):
            if subscripts == 'ij,jk->ik':
                return _backend_module.dot(operands[0], operands[1])
            elif subscripts == 'i,i->':
                return _backend_module.dot(operands[0], operands[1])
            else:
                raise NotImplementedError(f"einsum subscript '{subscripts}' not implemented for this backend")
    
    # Trigonometric functions
    if hasattr(_backend_module, 'arcsin'):
        arcsin = _backend_module.arcsin
        arccos = _backend_module.arccos
        arctan = _backend_module.arctan
        arctan2 = _backend_module.arctan2
    else:
        arcsin = _backend_module.asin
        arccos = _backend_module.acos
        arctan = _backend_module.atan
        arctan2 = _backend_module.atan2
    
    if hasattr(_backend_module, 'sinh'):
        sinh = _backend_module.sinh
        cosh = _backend_module.cosh
        tanh = _backend_module.tanh
    
    # Exponential and logarithmic functions
    if hasattr(_backend_module, 'exp2'):
        exp2 = _backend_module.exp2
    else:
        exp2 = lambda x: _backend_module.power(2, x)
    
    if hasattr(_backend_module, 'log2'):
        log2 = _backend_module.log2
    else:
        log2 = lambda x: _backend_module.log(x) / _backend_module.log(2)
    
    if hasattr(_backend_module, 'log10'):
        log10 = _backend_module.log10
    else:
        log10 = lambda x: _backend_module.log(x) / _backend_module.log(10)
    
    if hasattr(_backend_module, 'power'):
        power = _backend_module.power
    else:
        power = _backend_module.pow
    
    if hasattr(_backend_module, 'round'):
        round = _backend_module.round
    else:
        round = _backend_module.around
    
    if hasattr(_backend_module, 'angle'):
        angle = _backend_module.angle
    else:
        angle = lambda z: _backend_module.arctan2(_backend_module.imag(z), _backend_module.real(z))
    
    # Array conversion functions
    if hasattr(_backend_module, 'ascontiguousarray'):
        ascontiguousarray = _backend_module.ascontiguousarray
    else:
        ascontiguousarray = _backend_module.asarray
    
    if hasattr(_backend_module, 'asfortranarray'):
        asfortranarray = _backend_module.asfortranarray
    else:
        asfortranarray = _backend_module.asarray
    
    # Copy functions
    if hasattr(_backend_module, 'copy'):
        copy = _backend_module.copy
    else:
        copy = lambda a: _backend_module.array(a, copy=True)
    
    # Deep copy - use standard library
    import copy as copy_module
    deepcopy = copy_module.deepcopy
    
    # Shape and size functions
    shape = lambda x: x.shape
    if _BACKEND_NAME == "pytorch" or _BACKEND_NAME == "torch":
        size = lambda x: x.numel() if hasattr(x, 'numel') else _backend_module.numel(x)
    else:
        size = lambda x: x.size if hasattr(x, 'size') else _backend_module.size(x)
    
    # Type information functions
    if _BACKEND_NAME == "pytorch" or _BACKEND_NAME == "torch":
        # PyTorch doesn't have finfo, use numpy's
        import numpy
        finfo = numpy.finfo
    elif hasattr(_backend_module, 'finfo'):
        finfo = _backend_module.finfo
    else:
        import numpy
        finfo = numpy.finfo
    
    if hasattr(_backend_module, 'iinfo'):
        iinfo = _backend_module.iinfo
    else:
        import numpy
        iinfo = numpy.iinfo
    
    # Data types
    if _BACKEND_NAME == "pytorch" or _BACKEND_NAME == "torch":
        # PyTorch dtypes
        int8 = _backend_module.int8
        int16 = _backend_module.int16
        int32 = _backend_module.int32
        int64 = _backend_module.int64
        uint8 = _backend_module.uint8
        # PyTorch doesn't have all unsigned integer types
        uint16 = _backend_module.uint8  # Fallback
        uint32 = _backend_module.uint8  # Fallback
        uint64 = _backend_module.uint8  # Fallback
        float16 = _backend_module.float16
        float32 = _backend_module.float32
        float64 = _backend_module.float64
        complex64 = _backend_module.complex64
        complex128 = _backend_module.complex128
        # Add dtype function
        dtype = lambda x: x  # PyTorch dtypes are already objects
    elif hasattr(_backend_module, 'int8'):
        int8 = _backend_module.int8
        int16 = _backend_module.int16
        int32 = _backend_module.int32
        int64 = _backend_module.int64
        uint8 = _backend_module.uint8
        uint16 = _backend_module.uint16
        uint32 = _backend_module.uint32
        uint64 = _backend_module.uint64
        float16 = _backend_module.float16
        float32 = _backend_module.float32
        float64 = _backend_module.float64
        complex64 = _backend_module.complex64
        complex128 = _backend_module.complex128
        dtype = _backend_module.dtype
    else:
        import numpy
        int8 = numpy.int8
        int16 = numpy.int16
        int32 = numpy.int32
        int64 = numpy.int64
        uint8 = numpy.uint8
        uint16 = numpy.uint16
        uint32 = numpy.uint32
        uint64 = numpy.uint64
        float16 = numpy.float16
        float32 = numpy.float32
        float64 = numpy.float64
        complex64 = numpy.complex64
        complex128 = numpy.complex128
        dtype = numpy.dtype
    
    # Type hierarchies
    if hasattr(_backend_module, 'integer'):
        integer = _backend_module.integer
        floating = _backend_module.floating
        complexfloating = _backend_module.complexfloating
    else:
        import numpy
        integer = numpy.integer
        floating = numpy.floating
        complexfloating = numpy.complexfloating
    
    # Submodules
    if _BACKEND_NAME == "pytorch" or _BACKEND_NAME == "torch":
        # PyTorch has different random API
        class TorchRandomModule:
            @staticmethod
            def normal(loc=0.0, scale=1.0, size=None):
                if size is None:
                    return _backend_module.normal(loc, scale, (1,))
                elif isinstance(size, int):
                    return _backend_module.normal(loc, scale, (size,))
                else:
                    return _backend_module.normal(loc, scale, size)
            
            @staticmethod
            def rand(*args):
                if len(args) == 0:
                    return _backend_module.rand(1)
                return _backend_module.rand(*args)
            
            @staticmethod
            def randn(*args):
                if len(args) == 0:
                    return _backend_module.randn(1)
                return _backend_module.randn(*args)
            
            @staticmethod
            def randint(low, high=None, size=None, dtype=None):
                if high is None:
                    high = low
                    low = 0
                if size is None:
                    size = (1,)
                elif isinstance(size, int):
                    size = (size,)
                return _backend_module.randint(low, high, size, dtype=dtype)
            
            @staticmethod
            def uniform(low=0.0, high=1.0, size=None):
                if size is None:
                    size = (1,)
                elif isinstance(size, int):
                    size = (size,)
                return _backend_module.uniform(low, high, size)
        
        random = TorchRandomModule()
    elif hasattr(_backend_module, 'random'):
        random = _backend_module.random
    else:
        # Create a minimal random interface
        class RandomModule:
            @staticmethod
            def normal(loc=0.0, scale=1.0, size=None):
                if _BACKEND_NAME == "jax":
                    import jax.random as jax_random
                    key = jax_random.PRNGKey(0)  # Should be managed better in practice
                    return jax_random.normal(key, shape=size) * scale + loc
                else:
                    # Fallback to numpy random
                    import numpy
                    return numpy.random.normal(loc, scale, size)
            
            @staticmethod
            def rand(*args):
                if _BACKEND_NAME == "jax":
                    import jax.random as jax_random
                    key = jax_random.PRNGKey(0)
                    return jax_random.uniform(key, shape=args)
                else:
                    import numpy
                    return numpy.random.rand(*args)
            
            @staticmethod
            def randn(*args):
                if _BACKEND_NAME == "jax":
                    import jax.random as jax_random
                    key = jax_random.PRNGKey(0)
                    return jax_random.normal(key, shape=args)
                else:
                    import numpy
                    return numpy.random.randn(*args)
            
            @staticmethod
            def randint(low, high=None, size=None, dtype=int):
                if _BACKEND_NAME == "jax":
                    import jax.random as jax_random
                    key = jax_random.PRNGKey(0)
                    return jax_random.randint(key, shape=size or (), minval=low, maxval=high, dtype=dtype)
                else:
                    import numpy
                    return numpy.random.randint(low, high, size, dtype)
        
        random = RandomModule()
    
    if hasattr(_backend_module, 'linalg'):
        linalg = _backend_module.linalg
    else:
        # Create a minimal linalg interface
        class LinalgModule:
            @staticmethod
            def norm(x, ord=None, axis=None, keepdims=False):
                if ord is None or ord == 2:
                    return _backend_module.sqrt(_backend_module.sum(_backend_module.square(x), axis=axis, keepdims=keepdims))
                elif ord == 1:
                    return _backend_module.sum(_backend_module.abs(x), axis=axis, keepdims=keepdims)
                elif ord == _backend_module.inf:
                    return _backend_module.max(_backend_module.abs(x), axis=axis, keepdims=keepdims)
                else:
                    return _backend_module.power(_backend_module.sum(_backend_module.power(_backend_module.abs(x), ord), axis=axis, keepdims=keepdims), 1.0/ord)
            
            @staticmethod
            def solve(a, b):
                # This would need proper implementation for each backend
                if _BACKEND_NAME == "numpy":
                    import numpy
                    return xp.linalg.solve(a, b)
                else:
                    raise NotImplementedError(f"linalg.solve not implemented for backend {_BACKEND_NAME}")
            
            @staticmethod
            def inv(a):
                if _BACKEND_NAME == "numpy":
                    import numpy
                    return xp.linalg.inv(a)
                else:
                    raise NotImplementedError(f"linalg.inv not implemented for backend {_BACKEND_NAME}")
            
            @staticmethod
            def det(a):
                if _BACKEND_NAME == "numpy":
                    import numpy
                    return xp.linalg.det(a)
                else:
                    raise NotImplementedError(f"linalg.det not implemented for backend {_BACKEND_NAME}")
            
            @staticmethod
            def eig(a):
                if _BACKEND_NAME == "numpy":
                    import numpy
                    return xp.linalg.eig(a)
                else:
                    raise NotImplementedError(f"linalg.eig not implemented for backend {_BACKEND_NAME}")
            
            @staticmethod
            def eigvals(a):
                if _BACKEND_NAME == "numpy":
                    import numpy
                    return xp.linalg.eigvals(a)
                else:
                    raise NotImplementedError(f"linalg.eigvals not implemented for backend {_BACKEND_NAME}")
            
            @staticmethod
            def svd(a, full_matrices=True):
                if _BACKEND_NAME == "numpy":
                    import numpy
                    return xp.linalg.svd(a, full_matrices)
                else:
                    raise NotImplementedError(f"linalg.svd not implemented for backend {_BACKEND_NAME}")
        
        linalg = LinalgModule()
    
    if hasattr(_backend_module, 'fft'):
        fft = _backend_module.fft
    else:
        # Create a minimal fft interface
        class FFTModule:
            @staticmethod
            def fft(a, n=None, axis=-1, norm=None):
                if _BACKEND_NAME == "numpy":
                    import numpy
                    return xp.fft.fft(a, n, axis, norm)
                else:
                    raise NotImplementedError(f"fft.fft not implemented for backend {_BACKEND_NAME}")
            
            @staticmethod
            def ifft(a, n=None, axis=-1, norm=None):
                if _BACKEND_NAME == "numpy":
                    import numpy
                    return xp.fft.ifft(a, n, axis, norm)
                else:
                    raise NotImplementedError(f"fft.ifft not implemented for backend {_BACKEND_NAME}")
            
            @staticmethod
            def fftfreq(n, d=1.0):
                if _BACKEND_NAME == "numpy":
                    import numpy
                    return xp.fft.fftfreq(n, d)
                else:
                    raise NotImplementedError(f"fft.fftfreq not implemented for backend {_BACKEND_NAME}")
        
        fft = FFTModule()

# Initialize backend functions
_expose_backend_functions()

# Utility functions for backend compatibility
def is_array_like(obj):
    """Check if object is array-like for the current backend."""
    if _BACKEND_NAME == "numpy":
        import numpy
        return isinstance(obj, (xp.ndarray, list, tuple)) or xp.isscalar(obj)
    elif _BACKEND_NAME == "cupy":
        import cupy
        import numpy
        return isinstance(obj, (cupy.ndarray, xp.ndarray, list, tuple)) or xp.isscalar(obj)
    elif _BACKEND_NAME == "jax":
        import jax.numpy as jnp
        import numpy
        return isinstance(obj, (jnp.ndarray, xp.ndarray, list, tuple)) or xp.isscalar(obj)
    else:
        return hasattr(obj, '__array__') or hasattr(obj, '__array_interface__')

def to_numpy(arr):
    """Convert array to NumPy array regardless of backend."""
    if _BACKEND_NAME == "cupy":
        import cupy
        if isinstance(arr, cupy.ndarray):
            return cupy.asnumpy(arr)
    elif _BACKEND_NAME == "jax":
        import jax.numpy as jnp
        if isinstance(arr, jnp.ndarray):
            return jnp.asarray(arr).__array__()
    elif _BACKEND_NAME == "pytorch" or _BACKEND_NAME == "torch":
        import torch
        if isinstance(arr, torch.Tensor):
            return arr.detach().cpu().numpy()
    
    import numpy
    return numpy.asarray(arr)

def from_numpy(arr):
    """Convert NumPy array to current backend array."""
    if _BACKEND_NAME == "cupy":
        import cupy
        return cupy.asarray(arr)
    elif _BACKEND_NAME == "jax":
        import jax.numpy as jnp
        return jnp.asarray(arr)
    elif _BACKEND_NAME == "pytorch" or _BACKEND_NAME == "torch":
        import torch
        return torch.from_numpy(arr)
    else:
        return asarray(arr)

# Additional utility functions that might be needed
def meshgrid(*xi, **kwargs):
    """Create coordinate matrices from coordinate vectors."""
    if hasattr(_backend_module, 'meshgrid'):
        return _backend_module.meshgrid(*xi, **kwargs)
    else:
        # Basic meshgrid implementation
        indexing = kwargs.get('indexing', 'xy')
        if len(xi) == 2:
            x, y = xi
            if indexing == 'xy':
                X, Y = _backend_module.broadcast_arrays(x[:, None], y[None, :])
            else:  # 'ij'
                X, Y = _backend_module.broadcast_arrays(x[None, :], y[:, None])
            return X, Y
        else:
            raise NotImplementedError("meshgrid for more than 2 dimensions not implemented")

def broadcast_arrays(*arrays):
    """Broadcast arrays to a common shape."""
    if hasattr(_backend_module, 'broadcast_arrays'):
        return _backend_module.broadcast_arrays(*arrays)
    else:
        # Basic broadcasting - would need full implementation
        return arrays

def expand_dims(a, axis):
    """Expand the shape of an array."""
    if hasattr(_backend_module, 'expand_dims'):
        return _backend_module.expand_dims(a, axis)
    else:
        # Basic expand_dims implementation
        shape = list(a.shape)
        if axis < 0:
            axis = len(shape) + axis + 1
        shape.insert(axis, 1)
        return reshape(a, shape)

def squeeze(a, axis=None):
    """Remove single-dimensional entries from the shape of an array."""
    if hasattr(_backend_module, 'squeeze'):
        return _backend_module.squeeze(a, axis)
    else:
        # Basic squeeze implementation
        shape = list(a.shape)
        if axis is None:
            new_shape = [s for s in shape if s != 1]
        else:
            if isinstance(axis, int):
                axis = [axis]
            new_shape = [s for i, s in enumerate(shape) if i not in axis or s != 1]
        return reshape(a, new_shape)

# Additional mathematical functions that might be needed
def fix(x):
    """Round to nearest integer towards zero."""
    if hasattr(_backend_module, 'fix'):
        return _backend_module.fix(x)
    else:
        return _backend_module.trunc(x) if hasattr(_backend_module, 'trunc') else _backend_module.floor(x)

def isscalar(element):
    """Check if element is a scalar."""
    if hasattr(_backend_module, 'isscalar'):
        return _backend_module.isscalar(element)
    else:
        import numpy
        return xp.isscalar(element)

# Matrix operations that might be needed
def matrix(data, dtype=None):
    """Create a matrix from array-like input."""
    # Most backends don't have matrix type, so return regular array
    return asarray(data, dtype=dtype)

# Export the CUQIarray class as well
__all__ = [
    'CUQIarray', 'get_backend_name', 'set_backend', 'is_array_like', 'to_numpy', 'from_numpy',
    'array', 'zeros', 'ones', 'zeros_like', 'ones_like', 'empty', 'empty_like', 'full', 'full_like',
    'arange', 'linspace', 'logspace', 'eye', 'identity', 'diag', 'diagonal',
    'reshape', 'ravel', 'flatten', 'transpose', 'swapaxes', 'moveaxis',
    'concatenate', 'stack', 'vstack', 'hstack', 'dstack', 'split', 'hsplit', 'vsplit', 'dsplit',
    'sum', 'prod', 'mean', 'std', 'var', 'min', 'max', 'argmin', 'argmax', 'sort', 'argsort',
    'dot', 'matmul', 'inner', 'outer', 'cross', 'tensordot', 'einsum',
    'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan', 'arctan2', 'sinh', 'cosh', 'tanh',
    'exp', 'exp2', 'log', 'log2', 'log10', 'sqrt', 'square', 'power', 'abs', 'sign',
    'floor', 'ceil', 'round', 'clip', 'where', 'isnan', 'isinf', 'isfinite', 'count_nonzero', 'allclose', 'array_equiv', 'array_equal',
    'real', 'imag', 'conj', 'angle', 'absolute',
    'random', 'linalg', 'fft',
    'ndarray', 'dtype', 'newaxis', 'inf', 'nan', 'pi', 'e',
    'asarray', 'asanyarray', 'ascontiguousarray', 'asfortranarray',
    'copy', 'deepcopy', 'meshgrid', 'broadcast_arrays', 'expand_dims', 'squeeze',
    'fix', 'isscalar', 'matrix', 'finfo', 'iinfo', 'shape', 'size',
    'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64',
    'float16', 'float32', 'float64', 'complex64', 'complex128',
    'integer', 'floating', 'complexfloating'
]
