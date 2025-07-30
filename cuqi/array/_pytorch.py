"""
PyTorch backend for CUQIpy array operations.

This module provides a PyTorch interface for array operations.
Many operations raise NotImplementedError to focus on NumPy functionality.
"""

import warnings
import numpy as _np


def load_backend():
    """Load and return the PyTorch backend module."""
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError("PyTorch not available. Please install PyTorch to use the PyTorch backend.")


# =============================================================================
# Method Definitions
# =============================================================================

def _not_implemented(name):
    """Create a function that raises NotImplementedError for the given name."""
    def func(*args, **kwargs):
        raise NotImplementedError(f"{name} not implemented for PyTorch backend. Use NumPy backend for full functionality.")
    return func


def _convert_dtype_to_torch(dtype):
    """Convert numpy dtype to PyTorch dtype."""
    import torch as backend_module
    if dtype == _np.float64 or dtype == 'float64':
        return backend_module.float64
    elif dtype == _np.float32 or dtype == 'float32':
        return backend_module.float32
    elif dtype == _np.int64 or dtype == 'int64':
        return backend_module.int64
    elif dtype == _np.int32 or dtype == 'int32':
        return backend_module.int32
    elif dtype == _np.int16 or dtype == 'int16':
        return backend_module.int16
    elif dtype == _np.int8 or dtype == 'int8':
        return backend_module.int8
    elif dtype == _np.uint8 or dtype == 'uint8':
        return backend_module.uint8
    elif dtype == _np.bool_ or dtype == bool or dtype == 'bool':
        return backend_module.bool
    else:
        # Default fallback
        return backend_module.float64


def array_pytorch(x, dtype=None, requires_grad=False):
    """Create tensor with optional gradient tracking."""
    import torch as backend_module
    if x is None:
        return None
    if dtype is None:
        dtype = backend_module.float64
    else:
        # Convert numpy dtypes to PyTorch dtypes
        dtype = _convert_dtype_to_torch(dtype)
    return backend_module.tensor(x, dtype=dtype, requires_grad=requires_grad)


def zeros_pytorch(*args, dtype=None, **kwargs):
    """Create zeros tensor with default float64 dtype if not specified."""
    import torch as backend_module
    if dtype is None:
        dtype = backend_module.float64
    else:
        dtype = _convert_dtype_to_torch(dtype)
    return backend_module.zeros(*args, dtype=dtype, **kwargs)


def ones_pytorch(*args, dtype=None, **kwargs):
    """Create ones tensor with default float64 dtype if not specified."""
    import torch as backend_module
    if dtype is None:
        dtype = backend_module.float64
    else:
        dtype = _convert_dtype_to_torch(dtype)
    return backend_module.ones(*args, dtype=dtype, **kwargs)


def eye_pytorch(n, m=None, k=0, dtype=None):
    """Create identity matrix with optional offset."""
    import torch as backend_module
    if dtype is None:
        dtype = backend_module.float64
    else:
        dtype = _convert_dtype_to_torch(dtype)
    
    if m is None:
        m = n
    
    # Create base identity matrix
    if k == 0:
        return backend_module.eye(n, m, dtype=dtype)
    else:
        # Handle offset diagonal
        result = backend_module.zeros(n, m, dtype=dtype)
        if k > 0 and k < m:
            diag_size = min(n, m - k)
            result[:diag_size, k:k+diag_size] = backend_module.eye(diag_size, dtype=dtype)
        elif k < 0 and -k < n:
            diag_size = min(n + k, m)
            result[-k:-k+diag_size, :diag_size] = backend_module.eye(diag_size, dtype=dtype)
        return result


def identity_pytorch(n, dtype=None):
    """Create n x n identity matrix."""
    import torch as backend_module
    if dtype is None:
        dtype = backend_module.float64
    else:
        dtype = _convert_dtype_to_torch(dtype)
    return backend_module.eye(n, dtype=dtype)


def vstack_pytorch(tup):
    """Stack arrays vertically (row wise)."""
    import torch as backend_module
    # Convert all inputs to tensors
    tensors = []
    for arr in tup:
        if not isinstance(arr, backend_module.Tensor):
            arr = backend_module.tensor(arr)
        tensors.append(arr)
    return backend_module.vstack(tensors)


def hstack_pytorch(tup):
    """Stack arrays horizontally (column wise)."""
    import torch as backend_module
    # Convert all inputs to tensors
    tensors = []
    for arr in tup:
        if not isinstance(arr, backend_module.Tensor):
            arr = backend_module.tensor(arr)
        tensors.append(arr)
    return backend_module.hstack(tensors)


def dstack_pytorch(tup):
    """Stack arrays along the third axis (depth wise)."""
    import torch as backend_module
    # Convert all inputs to tensors
    tensors = []
    for arr in tup:
        if not isinstance(arr, backend_module.Tensor):
            arr = backend_module.tensor(arr)
        tensors.append(arr)
    return backend_module.dstack(tensors)


def max_pytorch(x, axis=None, keepdims=False):
    """Compute maximum along specified axis."""
    import torch as backend_module
    x = backend_module.tensor(x)
    if axis is None:
        return backend_module.max(x)
    else:
        result = backend_module.max(x, dim=axis, keepdim=keepdims)
        return result.values


def min_pytorch(x, axis=None, keepdims=False):
    """Compute minimum along specified axis."""
    import torch as backend_module
    x = backend_module.tensor(x)
    if axis is None:
        return backend_module.min(x)
    else:
        result = backend_module.min(x, dim=axis, keepdim=keepdims)
        return result.values


def piecewise_pytorch(x, condlist, funclist, *args, **kwargs):
    """Evaluate a piecewise-defined function."""
    import torch as backend_module
    x = backend_module.tensor(x)
    result = backend_module.zeros_like(x)
    
    for i, (cond, func) in enumerate(zip(condlist, funclist)):
        if callable(func):
            mask = backend_module.tensor(cond, dtype=backend_module.bool)
            result = backend_module.where(mask, func(x), result)
        else:
            mask = backend_module.tensor(cond, dtype=backend_module.bool)
            result = backend_module.where(mask, func, result)
    
    return result


def exp_pytorch(x):
    """Exponential function with scalar handling."""
    import torch as backend_module
    if backend_module.is_tensor(x):
        return backend_module.exp(x)
    else:
        return backend_module.exp(backend_module.tensor(x))


def log_pytorch(x):
    """Natural logarithm with scalar handling."""
    import torch as backend_module
    if backend_module.is_tensor(x):
        return backend_module.log(x)
    else:
        return backend_module.log(backend_module.tensor(x))


def log2_pytorch(x):
    """Base-2 logarithm with scalar handling."""
    import torch as backend_module
    if backend_module.is_tensor(x):
        return backend_module.log2(x)
    else:
        return backend_module.log2(backend_module.tensor(x))


def log10_pytorch(x):
    """Base-10 logarithm with scalar handling."""
    import torch as backend_module
    if backend_module.is_tensor(x):
        return backend_module.log10(x)
    else:
        return backend_module.log10(backend_module.tensor(x))


def sqrt_pytorch(x):
    """Square root with scalar handling."""
    import torch as backend_module
    if backend_module.is_tensor(x):
        return backend_module.sqrt(x)
    else:
        return backend_module.sqrt(backend_module.tensor(x))


def square_pytorch(x):
    """Square function with scalar handling."""
    import torch as backend_module
    if backend_module.is_tensor(x):
        return backend_module.square(x)
    else:
        return backend_module.square(backend_module.tensor(x))


def power_pytorch(x1, x2):
    """Power function with scalar handling."""
    import torch as backend_module
    if not backend_module.is_tensor(x1):
        x1 = backend_module.tensor(x1)
    if not backend_module.is_tensor(x2):
        x2 = backend_module.tensor(x2)
    return backend_module.pow(x1, x2)


def array_equiv_pytorch(a, b):
    """Check if arrays are equivalent (same shape and elements)."""
    import torch as backend_module
    
    # Handle None values
    if a is None and b is None:
        return True
    elif a is None or b is None:
        return False
    
    if not backend_module.is_tensor(a):
        a = backend_module.tensor(a)
    if not backend_module.is_tensor(b):
        b = backend_module.tensor(b)
    
    # Check shapes first
    if a.shape != b.shape:
        return False
    
    # Check if all elements are equal
    return backend_module.allclose(a, b)


def array_equal_pytorch(a, b):
    """Check if arrays are equal (same shape and elements, exact)."""
    import torch as backend_module
    
    # Handle None values
    if a is None and b is None:
        return True
    elif a is None or b is None:
        return False
    
    if not backend_module.is_tensor(a):
        a = backend_module.tensor(a)
    if not backend_module.is_tensor(b):
        b = backend_module.tensor(b)
    
    # Check shapes first
    if a.shape != b.shape:
        return False
    
    # Check if all elements are exactly equal
    return backend_module.all(backend_module.eq(a, b))


def isscalar_pytorch(element):
    """Check if element is a scalar."""
    import torch as backend_module
    if backend_module.is_tensor(element):
        # NumPy considers arrays (even 0-dim) as non-scalar, so match that behavior
        return False
    else:
        return _np.isscalar(element)


def shape_pytorch(x):
    """Get shape, handling sparse matrices properly."""
    import torch as backend_module
    if hasattr(x, 'shape') and hasattr(x, 'nnz'):  # sparse matrix
        return x.shape
    return tuple(backend_module.tensor(x).shape)


def squeeze_pytorch(x, axis=None):
    """Squeeze array dimensions."""
    import torch as backend_module
    x = backend_module.tensor(x)
    if axis is None:
        return backend_module.squeeze(x)
    else:
        return backend_module.squeeze(x, dim=axis)


def expand_dims_pytorch(x, axis):
    """Expand array dimensions."""
    import torch as backend_module
    x = backend_module.tensor(x)
    return backend_module.unsqueeze(x, dim=axis)


def transpose_pytorch(x, axes=None):
    """Transpose array dimensions."""
    import torch as backend_module
    x = backend_module.tensor(x)
    if axes is None:
        return x.T
    else:
        return x.permute(axes)


def sparse_spdiags_pytorch(data, diags, m, n, format=None):
    """Create sparse diagonal matrix using scipy.sparse.spdiags (fallback to NumPy)."""
    # Convert PyTorch tensors to NumPy for scipy.sparse operations
    if hasattr(data, 'cpu'):
        data = data.cpu().numpy()
    if hasattr(diags, 'cpu'):
        diags = diags.cpu().numpy()
    
    from scipy.sparse import spdiags
    return spdiags(data, diags, m, n, format=format)


def sparse_eye_pytorch(n, m=None, k=0, dtype=None, format=None):
    """Create sparse identity matrix using scipy.sparse.eye (fallback to NumPy)."""
    from scipy.sparse import eye
    if dtype is not None and hasattr(dtype, 'cpu'):
        dtype = dtype.cpu().numpy().dtype
    return eye(n, n=m, k=k, dtype=dtype, format=format)


def sparse_kron_pytorch(A, B, format=None):
    """Kronecker product of sparse matrices using scipy.sparse.kron (fallback to NumPy)."""
    from scipy.sparse import kron
    return kron(A, B, format=format)


def sparse_vstack_pytorch(blocks, format=None, dtype=None):
    """Stack sparse matrices vertically using scipy.sparse.vstack (fallback to NumPy)."""
    from scipy.sparse import vstack
    return vstack(blocks, format=format, dtype=dtype)


def issparse_pytorch(x):
    """Check if x is a sparse matrix."""
    from scipy.sparse import issparse
    # Import here to avoid circular imports
    from cuqi.array._sparse import BackendSparseMatrix
    return issparse(x) or isinstance(x, BackendSparseMatrix)


def _add_astype_to_tensor():
    """Add astype method to PyTorch tensors if not present."""
    import torch as backend_module
    if not hasattr(backend_module.Tensor, 'astype'):
        def astype(self, dtype):
            if dtype == float:
                return self.float()
            elif dtype == int:
                return self.int()
            else:
                return self.to(dtype)
        backend_module.Tensor.astype = astype


def to_numpy(arr):
    """Convert PyTorch tensor to NumPy array."""
    import torch
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    else:
        return _np.asarray(arr)


def dot_pytorch(a, b):
    """Dot product for PyTorch tensors, handling both 1D and 2D cases."""
    import torch as backend_module
    
    # Convert to tensors if needed
    if not isinstance(a, backend_module.Tensor):
        a = backend_module.tensor(a)
    if not isinstance(b, backend_module.Tensor):
        b = backend_module.tensor(b)
    
    # Handle different dimensionalities like NumPy
    if a.dim() == 1 and b.dim() == 1:
        # 1D dot product
        return backend_module.dot(a, b)
    elif a.dim() == 2 and b.dim() == 2:
        # 2D matrix multiplication
        return backend_module.matmul(a, b)
    elif a.dim() == 2 and b.dim() == 1:
        # Matrix-vector multiplication
        return backend_module.mv(a, b)
    elif a.dim() == 1 and b.dim() == 2:
        # Vector-matrix multiplication
        return backend_module.matmul(a.unsqueeze(0), b).squeeze(0)
    else:
        # For higher dimensions, use matmul
        return backend_module.matmul(a, b)


def copy_pytorch(x):
    """Backend-agnostic array copying for PyTorch."""
    # For PyTorch tensors, use .clone()
    if hasattr(x, 'clone'):
        return x.clone()
    # For NumPy arrays and matrices, use .copy()
    elif hasattr(x, 'copy'):
        return x.copy()
    # For other types, try to create a copy using the array constructor
    else:
        import copy as copy_module
        try:
            return copy_module.deepcopy(x)
        except:
            import torch as backend_module
            return backend_module.tensor(x)


def get_scipy_stats_pytorch():
    """Get scipy.stats module for PyTorch backend (same as NumPy)."""
    import scipy.stats
    return scipy.stats


def get_scipy_optimize_pytorch():
    """Get scipy.optimize module for PyTorch backend (same as NumPy)."""
    import scipy.optimize
    return scipy.optimize


def pad(array, pad_width, mode='constant', constant_values=0):
    """Pad a PyTorch tensor - basic implementation.
    
    This function is kept separate because it has a different signature
    and behavior compared to the standard array operations, and needs
    special handling for different backends.
    """
    import torch
    import torch.nn.functional as F
    
    if not isinstance(array, torch.Tensor):
        array = torch.tensor(array)
    
    # Convert numpy-style pad_width to PyTorch format
    if isinstance(pad_width, int):
        pad_width = [(pad_width, pad_width)]
    elif isinstance(pad_width, tuple) and len(pad_width) == 2 and isinstance(pad_width[0], int):
        pad_width = [pad_width]
    
    # PyTorch expects padding in reverse order and flattened
    torch_pad = []
    for pw in reversed(pad_width):
        if isinstance(pw, int):
            torch_pad.extend([pw, pw])
        else:
            torch_pad.extend([pw[0], pw[1]])
    
    if mode == 'constant':
        return F.pad(array, torch_pad, mode='constant', value=constant_values)
    else:
        return F.pad(array, torch_pad, mode=mode)


# =============================================================================
# Backend Functions Dictionary
# =============================================================================

def get_backend_functions(backend_module):
    """Get all array functions for PyTorch backend.
    
    Returns a dictionary of function names mapped to their implementations.
    Many functions raise NotImplementedError to focus on NumPy support.
    """
    functions = {}
    
    # Ensure we have PyTorch - no fallback
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch not available. Please install PyTorch to use the PyTorch backend.")
    
    # Array creation functions
    functions['array'] = array_pytorch
    functions['zeros'] = zeros_pytorch
    functions['ones'] = ones_pytorch
    functions['zeros_like'] = backend_module.zeros_like
    functions['ones_like'] = backend_module.ones_like
    functions['empty'] = backend_module.empty
    functions['empty_like'] = backend_module.empty_like
    functions['full'] = backend_module.full
    functions['full_like'] = backend_module.full_like
    functions['arange'] = backend_module.arange
    functions['linspace'] = backend_module.linspace
    functions['logspace'] = _not_implemented('logspace')
    functions['eye'] = eye_pytorch
    functions['identity'] = identity_pytorch
    functions['diag'] = backend_module.diag
    functions['diagonal'] = _not_implemented('diagonal')
    functions['meshgrid'] = _not_implemented('meshgrid')
    
    # Shape manipulation
    functions['reshape'] = backend_module.reshape
    functions['ravel'] = backend_module.ravel
    functions['flatten'] = lambda a: a.flatten()
    functions['transpose'] = transpose_pytorch
    functions['swapaxes'] = _not_implemented('swapaxes')
    functions['moveaxis'] = _not_implemented('moveaxis')
    functions['flip'] = backend_module.flip
    functions['flipud'] = _not_implemented('flipud')
    functions['fliplr'] = _not_implemented('fliplr')
    functions['rot90'] = _not_implemented('rot90')
    functions['roll'] = backend_module.roll
    
    # Array joining and splitting
    functions['concatenate'] = backend_module.cat
    functions['stack'] = backend_module.stack
    functions['vstack'] = vstack_pytorch
    functions['hstack'] = hstack_pytorch
    functions['dstack'] = dstack_pytorch
    functions['split'] = _not_implemented('split')
    functions['hsplit'] = _not_implemented('hsplit')
    functions['vsplit'] = _not_implemented('vsplit')
    functions['dsplit'] = _not_implemented('dsplit')
    
    # Mathematical functions
    functions['sum'] = backend_module.sum
    functions['prod'] = _not_implemented('prod')
    functions['mean'] = backend_module.mean
    functions['std'] = backend_module.std
    functions['var'] = backend_module.var
    functions['min'] = min_pytorch
    functions['max'] = max_pytorch
    functions['argmin'] = backend_module.argmin
    functions['argmax'] = backend_module.argmax
    functions['sort'] = backend_module.sort
    functions['argsort'] = backend_module.argsort
    functions['any'] = backend_module.any
    functions['all'] = backend_module.all
    functions['argwhere'] = _not_implemented('argwhere')
    functions['cumsum'] = backend_module.cumsum
    functions['cumprod'] = _not_implemented('cumprod')
    functions['diff'] = _not_implemented('diff')
    functions['gradient'] = _not_implemented('gradient')
    functions['maximum'] = backend_module.maximum
    functions['minimum'] = backend_module.minimum
    functions['repeat'] = _not_implemented('repeat')
    functions['isclose'] = backend_module.isclose
    functions['percentile'] = _not_implemented('percentile')
    functions['median'] = _not_implemented('median')
    functions['multiply'] = backend_module.multiply
    functions['tile'] = _not_implemented('tile')
    functions['float_power'] = _not_implemented('float_power')
    functions['piecewise'] = piecewise_pytorch
    
    # Linear algebra
    functions['dot'] = dot_pytorch
    functions['matmul'] = backend_module.matmul
    functions['inner'] = _not_implemented('inner')
    functions['outer'] = _not_implemented('outer')
    functions['cross'] = _not_implemented('cross')
    functions['tensordot'] = _not_implemented('tensordot')
    functions['einsum'] = _not_implemented('einsum')
    functions['tril'] = backend_module.tril
    functions['triu'] = backend_module.triu
    functions['linalg'] = backend_module.linalg
    
    # Trigonometric functions
    functions['sin'] = backend_module.sin
    functions['cos'] = backend_module.cos
    functions['tan'] = backend_module.tan
    functions['arcsin'] = backend_module.asin
    functions['arccos'] = backend_module.acos
    functions['arctan'] = backend_module.atan
    functions['arctan2'] = backend_module.atan2
    functions['sinh'] = backend_module.sinh
    functions['cosh'] = backend_module.cosh
    functions['tanh'] = backend_module.tanh
    
    # Exponential and logarithmic functions
    functions['exp'] = exp_pytorch
    functions['exp2'] = _not_implemented('exp2')
    functions['log'] = log_pytorch
    functions['log2'] = log2_pytorch
    functions['log10'] = log10_pytorch
    functions['sqrt'] = sqrt_pytorch
    functions['square'] = square_pytorch
    functions['power'] = power_pytorch
    functions['abs'] = backend_module.abs
    functions['sign'] = backend_module.sign
    
    # Rounding functions
    functions['floor'] = backend_module.floor
    functions['ceil'] = backend_module.ceil
    functions['round'] = backend_module.round
    functions['clip'] = backend_module.clamp
    
    # Logic functions
    functions['where'] = backend_module.where
    functions['isnan'] = backend_module.isnan
    functions['isinf'] = backend_module.isinf
    functions['isfinite'] = backend_module.isfinite
    functions['count_nonzero'] = backend_module.count_nonzero
    functions['allclose'] = backend_module.allclose
    functions['array_equiv'] = array_equiv_pytorch
    functions['array_equal'] = array_equal_pytorch
    functions['sinc'] = _not_implemented('sinc')
    functions['fix'] = backend_module.trunc
    
    # Complex numbers
    functions['real'] = backend_module.real
    functions['imag'] = backend_module.imag
    functions['conj'] = backend_module.conj
    functions['angle'] = _not_implemented('angle')
    functions['absolute'] = backend_module.abs
    
    # Array conversion
    functions['asarray'] = backend_module.as_tensor
    functions['asanyarray'] = backend_module.as_tensor
    functions['ascontiguousarray'] = _not_implemented('ascontiguousarray')
    functions['asfortranarray'] = _not_implemented('asfortranarray')
    functions['copy'] = copy_pytorch
    
    # Add astype method to tensors if not present
    _add_astype_to_tensor()
    
    # Data types and constants
    functions['finfo'] = _not_implemented('finfo')  # Use numpy's finfo
    functions['iinfo'] = _not_implemented('iinfo')  # Use numpy's iinfo
    functions['newaxis'] = None
    functions['inf'] = backend_module.inf
    functions['nan'] = backend_module.nan
    functions['pi'] = backend_module.pi
    functions['e'] = backend_module.e
    functions['size'] = lambda x: x.numel() if hasattr(x, 'numel') else backend_module.numel(x)
    functions['shape'] = shape_pytorch
    functions['squeeze'] = squeeze_pytorch
    functions['expand_dims'] = expand_dims_pytorch
    functions['equal'] = lambda x, y: backend_module.eq(backend_module.tensor(x), backend_module.tensor(y))
    functions['greater'] = lambda x, y: backend_module.gt(backend_module.tensor(x), backend_module.tensor(y))
    functions['less'] = lambda x, y: backend_module.lt(backend_module.tensor(x), backend_module.tensor(y))
    functions['ndim'] = lambda x: backend_module.tensor(x).ndim
    functions['isscalar'] = isscalar_pytorch
    
    # Data types
    functions['int8'] = backend_module.int8
    functions['int16'] = backend_module.int16
    functions['int32'] = backend_module.int32
    functions['int64'] = backend_module.int64
    functions['uint8'] = backend_module.uint8
    functions['uint16'] = _not_implemented('uint16')  # PyTorch doesn't have uint16
    functions['uint32'] = _not_implemented('uint32')  # PyTorch doesn't have uint32
    functions['uint64'] = _not_implemented('uint64')  # PyTorch doesn't have uint64
    functions['float16'] = backend_module.float16
    functions['float32'] = backend_module.float32
    functions['float64'] = backend_module.float64
    functions['complex64'] = backend_module.complex64
    functions['complex128'] = backend_module.complex128
    functions['bool_'] = backend_module.bool
    functions['ndarray'] = backend_module.Tensor
    functions['dtype'] = backend_module.dtype
    functions['integer'] = int  # Default integer type (use Python int)
    functions['floating'] = float  # Default floating type (use Python float)
    functions['complexfloating'] = complex  # Default complex type (use Python complex)
    
    # Modules - create mock modules for compatibility
    class _MockModule:
        def __getattr__(self, name):
            raise NotImplementedError(f"{name} not implemented for PyTorch backend. Use NumPy backend for full functionality.")
    
    functions['random'] = _MockModule()  # PyTorch has different random API
    functions['fft'] = _MockModule()
    functions['polynomial'] = _MockModule()
    functions['stats'] = get_scipy_stats_pytorch()
    functions['optimize'] = get_scipy_optimize_pytorch()
    
    # Sparse matrix functions
    functions['sparse_spdiags'] = sparse_spdiags_pytorch
    functions['sparse_eye'] = sparse_eye_pytorch
    functions['sparse_kron'] = sparse_kron_pytorch
    functions['sparse_vstack'] = sparse_vstack_pytorch
    functions['issparse'] = issparse_pytorch
    
    return functions