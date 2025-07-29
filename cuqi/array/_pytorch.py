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
        warnings.warn("PyTorch not available, falling back to NumPy")
        raise ImportError("PyTorch not available")


def get_backend_functions(backend_module):
    """Get all array functions for PyTorch backend.
    
    Returns a dictionary of function names mapped to their implementations.
    Many functions raise NotImplementedError to focus on NumPy support.
    """
    functions = {}
    
    # Check if we actually have PyTorch
    try:
        import torch
        is_torch = True
    except ImportError:
        is_torch = False
        # Fall back to numpy functions
        import numpy as np
        backend_module = np
    
    if not is_torch:
        # If PyTorch is not available, delegate to numpy backend
        from ._numpy import get_backend_functions as numpy_get_backend_functions
        return numpy_get_backend_functions(backend_module)
    
    # For PyTorch, implement minimal functionality and raise NotImplementedError for the rest
    def not_implemented(name):
        def func(*args, **kwargs):
            raise NotImplementedError(f"{name} not implemented for PyTorch backend. Use NumPy backend for full functionality.")
        return func
    
    # Array creation functions - basic implementation
    def array(x, dtype=None, requires_grad=False):
        """Create tensor with optional gradient tracking."""
        if x is None:
            return None
        if dtype is None:
            dtype = backend_module.float64
        return backend_module.tensor(x, dtype=dtype, requires_grad=requires_grad)
    
    def zeros(*args, dtype=None, **kwargs):
        if dtype is None:
            dtype = backend_module.float64
        return backend_module.zeros(*args, dtype=dtype, **kwargs)
    
    def ones(*args, dtype=None, **kwargs):
        if dtype is None:
            dtype = backend_module.float64
        return backend_module.ones(*args, dtype=dtype, **kwargs)
    
    functions['array'] = array
    functions['zeros'] = zeros
    functions['ones'] = ones
    functions['zeros_like'] = backend_module.zeros_like
    functions['ones_like'] = backend_module.ones_like
    functions['empty'] = backend_module.empty
    functions['empty_like'] = backend_module.empty_like
    functions['full'] = backend_module.full
    functions['full_like'] = backend_module.full_like
    functions['arange'] = backend_module.arange
    functions['linspace'] = backend_module.linspace
    
    # Many functions not implemented for PyTorch
    functions['logspace'] = not_implemented('logspace')
    functions['eye'] = backend_module.eye
    
    def identity_pytorch(n, dtype=None):
        """Create identity matrix for PyTorch."""
        if dtype is None:
            dtype = backend_module.float64
        return backend_module.eye(n, dtype=dtype)
    
    functions['identity'] = identity_pytorch
    functions['diag'] = backend_module.diag
    functions['diagonal'] = not_implemented('diagonal')
    functions['meshgrid'] = not_implemented('meshgrid')
    
    # Shape manipulation - basic support
    functions['reshape'] = backend_module.reshape
    functions['ravel'] = lambda a: a.flatten()
    functions['flatten'] = lambda a: a.flatten()
    functions['transpose'] = backend_module.transpose
    functions['swapaxes'] = not_implemented('swapaxes')
    functions['moveaxis'] = not_implemented('moveaxis')
    functions['flip'] = backend_module.flip
    functions['flipud'] = not_implemented('flipud')
    functions['fliplr'] = not_implemented('fliplr')
    functions['rot90'] = not_implemented('rot90')
    functions['roll'] = backend_module.roll
    
    # Array joining and splitting
    functions['concatenate'] = backend_module.cat  # PyTorch uses cat instead of concatenate
    functions['stack'] = backend_module.stack
    
    # Implement stack functions to match numpy API
    def vstack_pytorch(tup):
        """Vertical stack - stack arrays vertically (row-wise)."""
        # For 1D arrays, add a dimension and stack
        # For 2D+ arrays, concatenate along axis 0
        if len(tup) == 0:
            raise ValueError("Need at least one array to stack")
        
        first_arr = tup[0]
        if first_arr.dim() == 1:
            # For 1D arrays, we need to stack them to create a 2D array
            # Convert each 1D array to a row vector and stack
            expanded = [arr.unsqueeze(0) for arr in tup]
            return backend_module.cat(expanded, dim=0)
        else:
            # For 2D+ arrays, concatenate along axis 0
            return backend_module.cat(tup, dim=0)
    
    def hstack_pytorch(tup):
        """Horizontal stack - concatenate appropriately based on dimensions."""
        # For 1D arrays, concatenate along dim 0 (like NumPy)
        # For 2D+ arrays, concatenate along dim 1
        if len(tup) == 0:
            raise ValueError("Need at least one array to concatenate")
        
        first_arr = tup[0]
        if first_arr.dim() == 1:
            return backend_module.cat(tup, dim=0)
        else:
            return backend_module.cat(tup, dim=1)
    
    def dstack_pytorch(tup):
        """Depth stack - concatenate along axis 2."""
        # Expand dimensions if needed for 1D/2D arrays
        expanded = []
        for arr in tup:
            if arr.dim() == 1:
                # 1D -> 3D: (n,) -> (1, n, 1)
                expanded.append(arr.unsqueeze(0).unsqueeze(2))
            elif arr.dim() == 2:
                # 2D -> 3D: (m, n) -> (m, n, 1)
                expanded.append(arr.unsqueeze(2))
            else:
                expanded.append(arr)
        return backend_module.cat(expanded, dim=2)
    
    functions['vstack'] = vstack_pytorch
    functions['hstack'] = hstack_pytorch
    functions['dstack'] = dstack_pytorch
    functions['split'] = not_implemented('split')
    functions['hsplit'] = not_implemented('hsplit')
    functions['vsplit'] = not_implemented('vsplit')
    functions['dsplit'] = not_implemented('dsplit')
    
    # Mathematical functions - basic support
    functions['sum'] = backend_module.sum
    functions['prod'] = not_implemented('prod')
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
    functions['argwhere'] = not_implemented('argwhere')
    functions['cumsum'] = backend_module.cumsum
    functions['cumprod'] = not_implemented('cumprod')
    functions['diff'] = not_implemented('diff')
    functions['gradient'] = not_implemented('gradient')
    functions['maximum'] = backend_module.maximum
    functions['minimum'] = backend_module.minimum
    functions['repeat'] = not_implemented('repeat')
    functions['isclose'] = backend_module.isclose
    functions['percentile'] = not_implemented('percentile')
    functions['median'] = not_implemented('median')
    functions['multiply'] = backend_module.multiply
    functions['tile'] = not_implemented('tile')
    functions['float_power'] = not_implemented('float_power')
    def piecewise_pytorch(x, condlist, funclist, *args, **kwargs):
        """Piecewise function for PyTorch - convert to numpy, apply, convert back."""
        x_np = to_numpy(x)
        condlist_np = [to_numpy(c) for c in condlist]
        import numpy as np
        result_np = np.piecewise(x_np, condlist_np, funclist, *args, **kwargs)
        return backend_module.tensor(result_np, dtype=x.dtype if hasattr(x, 'dtype') else backend_module.float64)
    
    functions['piecewise'] = piecewise_pytorch
    
    # Linear algebra - basic support
    def dot(a, b):
        """Dot product - handle 1D and 2D cases."""
        if a.dim() == 1 and b.dim() == 1:
            return backend_module.dot(a, b)
        else:
            return backend_module.matmul(a, b)
    
    functions['dot'] = dot
    functions['matmul'] = backend_module.matmul
    functions['inner'] = not_implemented('inner')
    functions['outer'] = not_implemented('outer')
    functions['cross'] = not_implemented('cross')
    functions['tensordot'] = not_implemented('tensordot')
    functions['einsum'] = not_implemented('einsum')
    functions['tril'] = backend_module.tril
    functions['triu'] = backend_module.triu
    # PyTorch linalg module - create a minimal interface
    class TorchLinalgModule:
        @staticmethod
        def norm(x, ord=None, dim=None, keepdim=False):
            return backend_module.linalg.norm(x, ord=ord, dim=dim, keepdim=keepdim)
        
        @staticmethod
        def det(x):
            return backend_module.linalg.det(x)
        
        @staticmethod
        def inv(x):
            return backend_module.linalg.inv(x)
        
        @staticmethod
        def solve(a, b):
            return backend_module.linalg.solve(a, b)
        
        @staticmethod
        def eig(x):
            return backend_module.linalg.eig(x)
        
        @staticmethod
        def svd(x, full_matrices=True):
            return backend_module.linalg.svd(x, full_matrices=full_matrices)
    
    functions['linalg'] = TorchLinalgModule()
    
    # Trigonometric functions
    functions['sin'] = backend_module.sin
    functions['cos'] = backend_module.cos
    functions['tan'] = backend_module.tan
    functions['arcsin'] = backend_module.asin  # PyTorch uses asin instead of arcsin
    functions['arccos'] = backend_module.acos
    functions['arctan'] = backend_module.atan
    functions['arctan2'] = backend_module.atan2
    functions['sinh'] = backend_module.sinh
    functions['cosh'] = backend_module.cosh
    functions['tanh'] = backend_module.tanh
    
    # Exponential and logarithmic functions
    def exp_pytorch(x):
        if not isinstance(x, backend_module.Tensor):
            x = backend_module.tensor(x)
        return backend_module.exp(x)
    
    def log_pytorch(x):
        if not isinstance(x, backend_module.Tensor):
            x = backend_module.tensor(x)
        return backend_module.log(x)
    
    def log2_pytorch(x):
        if not isinstance(x, backend_module.Tensor):
            x = backend_module.tensor(x)
        return backend_module.log2(x)
    
    def log10_pytorch(x):
        if not isinstance(x, backend_module.Tensor):
            x = backend_module.tensor(x)
        return backend_module.log10(x)
    
    functions['exp'] = exp_pytorch
    functions['exp2'] = not_implemented('exp2')
    functions['log'] = log_pytorch
    functions['log2'] = log2_pytorch
    functions['log10'] = log10_pytorch
    def sqrt_pytorch(x):
        if not isinstance(x, backend_module.Tensor):
            x = backend_module.tensor(x)
        return backend_module.sqrt(x)
    
    def square_pytorch(x):
        if not isinstance(x, backend_module.Tensor):
            x = backend_module.tensor(x)
        return backend_module.square(x)
    
    def power_pytorch(x1, x2):
        if not isinstance(x1, backend_module.Tensor):
            x1 = backend_module.tensor(x1)
        if not isinstance(x2, backend_module.Tensor):
            x2 = backend_module.tensor(x2)
        return backend_module.pow(x1, x2)
    
    functions['sqrt'] = sqrt_pytorch
    functions['square'] = square_pytorch
    functions['power'] = power_pytorch  # PyTorch uses pow instead of power
    functions['abs'] = backend_module.abs
    functions['sign'] = backend_module.sign
    
    # Rounding functions
    functions['floor'] = backend_module.floor
    functions['ceil'] = backend_module.ceil
    functions['round'] = backend_module.round
    functions['clip'] = backend_module.clamp  # PyTorch uses clamp instead of clip
    
    # Logic functions
    functions['where'] = backend_module.where
    functions['isnan'] = backend_module.isnan
    functions['isinf'] = backend_module.isinf
    functions['isfinite'] = backend_module.isfinite
    functions['count_nonzero'] = backend_module.count_nonzero
    functions['allclose'] = backend_module.allclose
    
    def array_equiv_pytorch(a, b):
        """Check if arrays are equivalent for PyTorch."""
        if a is None and b is None:
            return True
        elif a is None or b is None:
            return False
        
        # Convert to tensors if needed
        if not isinstance(a, backend_module.Tensor):
            a = backend_module.tensor(a)
        if not isinstance(b, backend_module.Tensor):
            b = backend_module.tensor(b)
        
        # Check if shapes are compatible for broadcasting
        try:
            result = backend_module.equal(a, b)
            return result.all().item()
        except:
            return False
    
    def array_equal_pytorch(a, b):
        """Check if arrays are exactly equal for PyTorch."""
        if a is None and b is None:
            return True
        elif a is None or b is None:
            return False
        
        # Convert to tensors if needed
        if not isinstance(a, backend_module.Tensor):
            a = backend_module.tensor(a)
        if not isinstance(b, backend_module.Tensor):
            b = backend_module.tensor(b)
        
        # Check exact equality (same shape and values)
        if a.shape != b.shape:
            return False
        
        result = backend_module.equal(a, b)
        return result.all().item()
    
    functions['array_equiv'] = array_equiv_pytorch
    functions['array_equal'] = array_equal_pytorch
    def isscalar_pytorch(element):
        """Check if element is a scalar for PyTorch backend."""
        import numpy as np
        # Check if we have actual PyTorch tensors
        if hasattr(backend_module, 'Tensor') and isinstance(element, backend_module.Tensor):
            return element.dim() == 0
        return np.isscalar(element)
    
    functions['isscalar'] = isscalar_pytorch
    functions['sinc'] = not_implemented('sinc')
    functions['fix'] = backend_module.trunc  # PyTorch uses trunc for fix
    
    # Complex numbers
    functions['real'] = backend_module.real
    functions['imag'] = backend_module.imag
    functions['conj'] = backend_module.conj
    functions['angle'] = not_implemented('angle')
    functions['absolute'] = backend_module.abs
    
    # Array conversion
    functions['asarray'] = lambda x, dtype=None: backend_module.tensor(x, dtype=dtype) if not isinstance(x, backend_module.Tensor) else x
    functions['asanyarray'] = lambda x, dtype=None: backend_module.tensor(x, dtype=dtype) if not isinstance(x, backend_module.Tensor) else x
    functions['ascontiguousarray'] = not_implemented('ascontiguousarray')
    functions['asfortranarray'] = not_implemented('asfortranarray')
    functions['copy'] = lambda x: x.clone()
    
    # Add astype method to tensors if not present
    def add_astype_to_tensor():
        if not hasattr(backend_module.Tensor, 'astype'):
            def astype(self, dtype):
                if dtype == float:
                    return self.float()
                elif dtype == int:
                    return self.int()
                else:
                    return self.to(dtype)
            backend_module.Tensor.astype = astype
    
    add_astype_to_tensor()
    
    # Data types and constants
    functions['finfo'] = not_implemented('finfo')  # Use numpy's finfo
    functions['iinfo'] = not_implemented('iinfo')  # Use numpy's iinfo
    functions['newaxis'] = None  # PyTorch uses None for new axis
    functions['inf'] = float('inf')
    functions['nan'] = float('nan')
    import math
    functions['pi'] = math.pi
    functions['e'] = math.e
    functions['size'] = lambda x: x.numel() if hasattr(x, 'numel') else backend_module.numel(x)
    functions['shape'] = lambda x: x.shape
    # Use numpy integer types for better compatibility
    import numpy as np
    functions['int8'] = np.int8
    functions['int16'] = np.int16
    functions['int32'] = np.int32
    functions['int64'] = np.int64
    functions['uint8'] = np.uint8
    functions['uint16'] = np.uint16
    functions['uint32'] = np.uint32
    functions['uint64'] = np.uint64
    functions['float16'] = backend_module.float16
    functions['float32'] = backend_module.float32
    functions['float64'] = backend_module.float64
    functions['complex64'] = backend_module.complex64
    functions['complex128'] = backend_module.complex128
    functions['bool_'] = backend_module.bool
    functions['ndarray'] = backend_module.Tensor  # PyTorch uses Tensor class
    functions['dtype'] = lambda x: x  # PyTorch dtypes are already objects
    # Use numpy type hierarchies for compatibility
    import numpy as np
    functions['integer'] = np.integer
    functions['floating'] = np.floating
    functions['complexfloating'] = np.complexfloating
    
    # Modules - create minimal interfaces for PyTorch
    class TorchRandomModule:
        @staticmethod
        def normal(loc=0.0, scale=1.0, size=None):
            if size is None:
                return backend_module.normal(loc, scale, (1,))
            elif isinstance(size, int):
                return backend_module.normal(loc, scale, (size,))
            else:
                return backend_module.normal(loc, scale, size)
        
        @staticmethod
        def rand(*args):
            if len(args) == 0:
                return backend_module.rand(1)
            return backend_module.rand(*args)
        
        @staticmethod
        def randn(*args):
            if len(args) == 0:
                return backend_module.randn(1)
            return backend_module.randn(*args)
        
        @staticmethod
        def randint(low, high=None, size=None, dtype=None):
            if high is None:
                high = low
                low = 0
            if size is None:
                size = (1,)
            elif isinstance(size, int):
                size = (size,)
            return backend_module.randint(low, high, size, dtype=dtype)
        
        @staticmethod
        def uniform(low=0.0, high=1.0, size=None):
            if size is None:
                size = (1,)
            elif isinstance(size, int):
                size = (size,)
            return backend_module.uniform(low, high, size)
        
        @staticmethod
        def default_rng(seed=None):
            """Default random number generator - not implemented for PyTorch."""
            raise NotImplementedError("random.default_rng not implemented for PyTorch backend. Use NumPy backend for full functionality.")
    
    functions['random'] = TorchRandomModule()
    functions['fft'] = not_implemented('fft')
    
    class TorchPolynomialModule:
        class legendre:
            @staticmethod
            def leggauss(deg):
                """Gauss-Legendre quadrature - not implemented for PyTorch."""
                raise NotImplementedError("polynomial.legendre.leggauss not implemented for PyTorch backend. Use NumPy backend for full functionality.")
    
    functions['polynomial'] = TorchPolynomialModule()
    
    # Sparse matrix functions using PyTorch sparse tensors
    def sparse_spdiags_pytorch(data, diags, m, n, format=None):
        """Create sparse diagonal matrix using PyTorch sparse tensors."""
        # Convert PyTorch tensors to NumPy for scipy, then convert result back if needed
        data_np = data.detach().cpu().numpy() if isinstance(data, backend_module.Tensor) else data
        diags_np = diags if isinstance(diags, (list, tuple)) else diags.detach().cpu().numpy()
        
        from scipy.sparse import spdiags
        sparse_matrix = spdiags(data_np, diags_np, m, n, format=format)
        
        # For now, return the scipy sparse matrix - PyTorch can work with it
        return sparse_matrix
    
    def sparse_eye_pytorch(n, m=None, k=0, dtype=None, format=None):
        """Create sparse identity matrix."""
        if dtype is None:
            dtype = backend_module.float64
        
        from scipy.sparse import eye
        return eye(n, m=m, k=k, dtype=_np.float64, format=format)
    
    def sparse_kron_pytorch(A, B, format=None):
        """Kronecker product of sparse matrices."""
        from scipy.sparse import kron
        return kron(A, B, format=format)
    
    def sparse_vstack_pytorch(blocks, format=None, dtype=None):
        """Stack sparse matrices vertically."""
        from scipy.sparse import vstack
        return vstack(blocks, format=format, dtype=dtype)
    
    functions['sparse_spdiags'] = sparse_spdiags_pytorch
    functions['sparse_eye'] = sparse_eye_pytorch
    functions['sparse_kron'] = sparse_kron_pytorch
    functions['sparse_vstack'] = sparse_vstack_pytorch
    
    return functions


def to_numpy(arr):
    """Convert PyTorch tensor to NumPy array."""
    try:
        import torch
        if isinstance(arr, torch.Tensor):
            return arr.detach().cpu().numpy()
    except ImportError:
        pass
    
    import numpy as np
    return np.asarray(arr)


def pad(array, pad_width, mode='constant', constant_values=0):
    """Pad a PyTorch tensor - basic implementation."""
    try:
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
    except ImportError:
        # Fall back to numpy
        import numpy as np
        arr_np = to_numpy(array)
        if mode == 'constant':
            return np.pad(arr_np, pad_width, mode=mode, constant_values=constant_values)
        else:
            return np.pad(arr_np, pad_width, mode=mode)