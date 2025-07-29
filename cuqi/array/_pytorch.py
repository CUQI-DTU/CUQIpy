"""
PyTorch backend for CUQIpy array operations.

This module provides a PyTorch interface for array operations.
Many operations raise NotImplementedError to focus on NumPy functionality.
"""

import warnings


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
    functions['identity'] = not_implemented('identity')
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
    functions['vstack'] = not_implemented('vstack')
    functions['hstack'] = not_implemented('hstack')
    functions['dstack'] = not_implemented('dstack')
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
    functions['exp'] = backend_module.exp
    functions['exp2'] = not_implemented('exp2')
    functions['log'] = backend_module.log
    functions['log2'] = backend_module.log2
    functions['log10'] = backend_module.log10
    functions['sqrt'] = backend_module.sqrt
    functions['square'] = backend_module.square
    functions['power'] = backend_module.pow  # PyTorch uses pow instead of power
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
    functions['array_equiv'] = not_implemented('array_equiv')
    functions['array_equal'] = not_implemented('array_equal')
    functions['isscalar'] = not_implemented('isscalar')
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
    functions['int8'] = backend_module.int8
    functions['int16'] = backend_module.int16
    functions['int32'] = backend_module.int32
    functions['int64'] = backend_module.int64
    functions['uint8'] = backend_module.uint8
    functions['uint16'] = backend_module.uint8  # Fallback
    functions['uint32'] = backend_module.uint8  # Fallback
    functions['uint64'] = backend_module.uint8  # Fallback
    functions['float16'] = backend_module.float16
    functions['float32'] = backend_module.float32
    functions['float64'] = backend_module.float64
    functions['complex64'] = backend_module.complex64
    functions['complex128'] = backend_module.complex128
    functions['bool_'] = backend_module.bool
    functions['ndarray'] = backend_module.Tensor  # PyTorch uses Tensor class
    functions['dtype'] = lambda x: x  # PyTorch dtypes are already objects
    functions['integer'] = not_implemented('integer')
    functions['floating'] = not_implemented('floating')
    functions['complexfloating'] = not_implemented('complexfloating')
    
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