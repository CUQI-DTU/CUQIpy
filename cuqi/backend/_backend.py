import importlib

_backend = "numpy"
_backend_mod = importlib.import_module(_backend)

def set(backend_name):
    global _backend, _backend_mod
    if backend_name == "torch":
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is not installed. Please install it to use the torch backend.")
        _backend = "torch"
        _backend_mod = torch
    elif backend_name == "numpy":
        import numpy
        _backend = "numpy"
        _backend_mod = numpy
    else:
        raise ValueError(f"Unsupported backend: {backend_name}")

def array(*args, **kwargs):
    if _backend == "torch":
        return _backend_mod.tensor(*args, **kwargs)
    else:
        return _backend_mod.array(*args, **kwargs)

def zeros(*args, **kwargs):
    if _backend == "torch":
        return _backend_mod.zeros(*args, **kwargs)
    else:
        return _backend_mod.zeros(*args, **kwargs)

def asarray(*args, **kwargs):
    if _backend == "torch":
        return _backend_mod.tensor(*args, **kwargs)
    else:
        return _backend_mod.asarray(*args, **kwargs)

def is_array(obj):
    if _backend == "numpy":
        import numpy
        return isinstance(obj, numpy.ndarray)
    elif _backend == "torch":
        return _backend_mod.is_tensor(obj)
    else:
        return False

def as_numpy(obj):
    """Convert any array to numpy array for external solvers"""
    if _backend == "numpy":
        import numpy
        return numpy.array(obj)
    elif _backend == "torch":
        if hasattr(obj, 'cpu') and hasattr(obj, 'numpy'):
            return obj.cpu().numpy()
        else:
            import numpy
            return numpy.array(obj)
    else:
        import numpy
        return numpy.array(obj) 