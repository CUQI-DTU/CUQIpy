"""
NumPy backend for CUQIpy array operations.

This module provides a clean interface to NumPy array operations,
maintaining full compatibility with existing CUQI code.
"""

import numpy as _np
import warnings


def load_backend():
    """Load and return the NumPy backend module."""
    return _np


def get_backend_functions(backend_module):
    """Get all array functions for NumPy backend.
    
    Returns a dictionary of function names mapped to their implementations.
    """
    # For NumPy, we can use most functions directly
    functions = {}
    
    # Array creation functions
    functions['array'] = lambda x, dtype=None: backend_module.array(x, dtype=dtype)
    def zeros_with_default_dtype(*args, dtype=None, **kwargs):
        if dtype is None:
            dtype = backend_module.float64
        return backend_module.zeros(*args, dtype=dtype, **kwargs)
    
    def ones_with_default_dtype(*args, dtype=None, **kwargs):
        if dtype is None:
            dtype = backend_module.float64
        return backend_module.ones(*args, dtype=dtype, **kwargs)
    
    functions['zeros'] = zeros_with_default_dtype
    functions['ones'] = ones_with_default_dtype
    functions['zeros_like'] = backend_module.zeros_like
    functions['ones_like'] = backend_module.ones_like
    functions['empty'] = backend_module.empty
    functions['empty_like'] = backend_module.empty_like
    functions['full'] = backend_module.full
    functions['full_like'] = backend_module.full_like
    functions['arange'] = backend_module.arange
    functions['linspace'] = backend_module.linspace
    functions['logspace'] = backend_module.logspace
    functions['eye'] = backend_module.eye
    functions['identity'] = backend_module.identity
    functions['diag'] = backend_module.diag
    functions['diagonal'] = backend_module.diagonal
    functions['meshgrid'] = backend_module.meshgrid
    
    # Shape manipulation
    functions['reshape'] = backend_module.reshape
    functions['ravel'] = backend_module.ravel
    functions['flatten'] = lambda a: a.flatten()
    functions['transpose'] = backend_module.transpose
    functions['swapaxes'] = backend_module.swapaxes
    functions['moveaxis'] = backend_module.moveaxis
    functions['flip'] = backend_module.flip
    functions['flipud'] = backend_module.flipud
    functions['fliplr'] = backend_module.fliplr
    functions['rot90'] = backend_module.rot90
    functions['roll'] = backend_module.roll
    
    # Array joining and splitting
    functions['concatenate'] = backend_module.concatenate
    functions['stack'] = backend_module.stack
    functions['vstack'] = backend_module.vstack
    functions['hstack'] = backend_module.hstack
    functions['dstack'] = backend_module.dstack
    functions['split'] = backend_module.split
    functions['hsplit'] = backend_module.hsplit
    functions['vsplit'] = backend_module.vsplit
    functions['dsplit'] = backend_module.dsplit
    
    # Mathematical functions
    functions['sum'] = backend_module.sum
    functions['prod'] = backend_module.prod
    functions['mean'] = backend_module.mean
    functions['std'] = backend_module.std
    functions['var'] = backend_module.var
    functions['min'] = backend_module.min
    functions['max'] = backend_module.max
    functions['argmin'] = backend_module.argmin
    functions['argmax'] = backend_module.argmax
    functions['sort'] = backend_module.sort
    functions['argsort'] = backend_module.argsort
    functions['any'] = backend_module.any
    functions['all'] = backend_module.all
    functions['argwhere'] = backend_module.argwhere
    functions['cumsum'] = backend_module.cumsum
    functions['cumprod'] = backend_module.cumprod
    functions['diff'] = backend_module.diff
    functions['gradient'] = backend_module.gradient
    functions['maximum'] = backend_module.maximum
    functions['minimum'] = backend_module.minimum
    functions['repeat'] = backend_module.repeat
    functions['isclose'] = backend_module.isclose
    functions['percentile'] = backend_module.percentile
    functions['median'] = backend_module.median
    functions['multiply'] = backend_module.multiply
    functions['tile'] = backend_module.tile
    functions['float_power'] = backend_module.float_power
    functions['piecewise'] = backend_module.piecewise
    
    # Linear algebra
    functions['dot'] = backend_module.dot
    functions['matmul'] = backend_module.matmul
    functions['inner'] = backend_module.inner
    functions['outer'] = backend_module.outer
    functions['cross'] = backend_module.cross
    functions['tensordot'] = backend_module.tensordot
    functions['einsum'] = backend_module.einsum
    functions['tril'] = backend_module.tril
    functions['triu'] = backend_module.triu
    functions['linalg'] = backend_module.linalg
    
    # Trigonometric functions
    functions['sin'] = backend_module.sin
    functions['cos'] = backend_module.cos
    functions['tan'] = backend_module.tan
    functions['arcsin'] = backend_module.arcsin
    functions['arccos'] = backend_module.arccos
    functions['arctan'] = backend_module.arctan
    functions['arctan2'] = backend_module.arctan2
    functions['sinh'] = backend_module.sinh
    functions['cosh'] = backend_module.cosh
    functions['tanh'] = backend_module.tanh
    
    # Exponential and logarithmic functions
    functions['exp'] = backend_module.exp
    functions['exp2'] = backend_module.exp2
    functions['log'] = backend_module.log
    functions['log2'] = backend_module.log2
    functions['log10'] = backend_module.log10
    functions['sqrt'] = backend_module.sqrt
    functions['square'] = backend_module.square
    functions['power'] = backend_module.power
    functions['abs'] = backend_module.abs
    functions['sign'] = backend_module.sign
    
    # Rounding functions
    functions['floor'] = backend_module.floor
    functions['ceil'] = backend_module.ceil
    functions['round'] = backend_module.round
    functions['clip'] = backend_module.clip
    
    # Logic functions
    functions['where'] = backend_module.where
    functions['isnan'] = backend_module.isnan
    functions['isinf'] = backend_module.isinf
    functions['isfinite'] = backend_module.isfinite
    functions['count_nonzero'] = backend_module.count_nonzero
    functions['allclose'] = backend_module.allclose
    functions['array_equiv'] = backend_module.array_equiv
    functions['array_equal'] = backend_module.array_equal
    functions['isscalar'] = backend_module.isscalar
    functions['sinc'] = backend_module.sinc
    functions['fix'] = backend_module.fix
    
    # Complex numbers
    functions['real'] = backend_module.real
    functions['imag'] = backend_module.imag
    functions['conj'] = backend_module.conj
    functions['angle'] = backend_module.angle
    functions['absolute'] = backend_module.absolute
    
    # Array conversion
    functions['asarray'] = backend_module.asarray
    functions['asanyarray'] = backend_module.asanyarray
    functions['ascontiguousarray'] = backend_module.ascontiguousarray
    functions['asfortranarray'] = backend_module.asfortranarray
    functions['copy'] = backend_module.copy
    
    # Data types and constants
    functions['finfo'] = backend_module.finfo
    functions['iinfo'] = backend_module.iinfo
    functions['newaxis'] = backend_module.newaxis
    functions['inf'] = backend_module.inf
    functions['nan'] = backend_module.nan
    functions['pi'] = backend_module.pi
    functions['e'] = backend_module.e
    functions['size'] = backend_module.size
    functions['shape'] = lambda x: x.shape
    functions['int8'] = backend_module.int8
    functions['int16'] = backend_module.int16
    functions['int32'] = backend_module.int32
    functions['int64'] = backend_module.int64
    functions['uint8'] = backend_module.uint8
    functions['uint16'] = backend_module.uint16
    functions['uint32'] = backend_module.uint32
    functions['uint64'] = backend_module.uint64
    functions['float16'] = backend_module.float16
    functions['float32'] = backend_module.float32
    functions['float64'] = backend_module.float64
    functions['complex64'] = backend_module.complex64
    functions['complex128'] = backend_module.complex128
    functions['bool_'] = backend_module.bool_
    functions['ndarray'] = backend_module.ndarray
    functions['dtype'] = backend_module.dtype
    functions['integer'] = backend_module.integer
    functions['floating'] = backend_module.floating
    functions['complexfloating'] = backend_module.complexfloating
    
    # Modules
    functions['random'] = backend_module.random
    functions['fft'] = backend_module.fft
    functions['polynomial'] = backend_module.polynomial
    
    return functions


def to_numpy(arr):
    """Convert array to NumPy array (no-op for NumPy backend)."""
    return _np.asarray(arr)


def pad(array, pad_width, mode='constant', constant_values=0):
    """Pad an array using NumPy's pad function."""
    if mode == 'constant':
        return _np.pad(array, pad_width, mode=mode, constant_values=constant_values)
    else:
        return _np.pad(array, pad_width, mode=mode)

# Sparse matrix functions for NumPy backend
def sparse_spdiags(data, diags, m, n, format=None):
    """Create sparse diagonal matrix using scipy.sparse.spdiags."""
    from scipy.sparse import spdiags
    return spdiags(data, diags, m, n, format=format)

def sparse_eye(n, m=None, k=0, dtype=float, format=None):
    """Create sparse identity matrix using scipy.sparse.eye."""
    from scipy.sparse import eye
    return eye(n, m=m, k=k, dtype=dtype, format=format)

def sparse_kron(A, B, format=None):
    """Kronecker product of sparse matrices using scipy.sparse.kron."""
    from scipy.sparse import kron
    return kron(A, B, format=format)

def sparse_vstack(blocks, format=None, dtype=None):
    """Stack sparse matrices vertically using scipy.sparse.vstack."""
    from scipy.sparse import vstack
    return vstack(blocks, format=format, dtype=dtype)