from cuqi.geometry import _DefaultGeometry1D
import cuqi.backend as xp

class CUQIarray:
    """
    A class to represent data arrays, agnostic to backend, along with geometry and plotting

    Parameters
    ----------
    input_array : array-like
        An array holding the parameter or function values (NumPy, Torch, etc.).
    is_par : bool, default True
        Boolean flag whether input_array is to be interpreted as parameter (True) or function values (False).
    geometry : cuqi.geometry.Geometry, default None
        Contains the geometry related of the data.
    """
    def __init__(self, input_array, is_par=True, geometry=None):
        self._array = xp.asarray(input_array)
        self.is_par = is_par
        
        # Check for multidimensional arrays when is_par=True
        if is_par and self._array.ndim > 1:
            raise ValueError("input_array cannot be multidimensional when initializing CUQIarray as parameter (with is_par True).")
        
        if (not is_par) and (geometry is None):
            raise ValueError("geometry cannot be none when initializing a CUQIarray as function values (with is_par False).")
        if geometry is None:
            # Get size in a backend-agnostic way
            if hasattr(self._array, 'numel'):
                # PyTorch tensor
                size = self._array.numel()
            elif hasattr(self._array, 'size') and not callable(getattr(self._array, 'size', None)):
                # NumPy array with size property
                size = self._array.size
            else:
                size = len(self._array)
            # Create a grid for the geometry (convert to numpy for compatibility)
            import numpy
            grid = numpy.arange(size)
            self.geometry = _DefaultGeometry1D(grid)
        else:
            self.geometry = geometry

    def __array__(self, dtype=None):
        """Convert to numpy array for compatibility with numpy operations."""
        result = xp.as_numpy(self._array)
        if dtype is not None:
            result = result.astype(dtype)
        return result

    @property
    def shape(self):
        return self._array.shape

    @property
    def size(self):
        if hasattr(self._array, 'numel'):
            # PyTorch tensor
            return self._array.numel()
        elif hasattr(self._array, 'size') and not callable(getattr(self._array, 'size', None)):
            # NumPy array with size property
            return self._array.size
        else:
            return len(self._array)

    @property
    def ndim(self):
        return self._array.ndim

    @property
    def dtype(self):
        return self._array.dtype

    @property
    def array(self):
        """Return the underlying array (for backward compatibility)."""
        return self._array

    @property
    def funvals(self):
        """ Returns itself as function values. """
        if self.is_par is True:
            vals = self.geometry.par2fun(self._array)
        else:
            vals = self._array
        # Return the underlying array, not wrapped in CUQIarray
        return vals

    @property
    def parameters(self):
        """ Returns itself as parameters. """
        if self.is_par is False:
            vals = self.geometry.fun2par(self._array)
            return CUQIarray(vals, is_par=True, geometry=self.geometry)
        else:
            return self

    def as_numpy(self):
        """Convert to numpy array for external solvers."""
        return xp.as_numpy(self._array)

    def to_numpy(self):
        """Convert to numpy array."""
        return xp.as_numpy(self._array)

    def to_torch(self):
        """Convert to torch tensor."""
        import torch
        if hasattr(self._array, 'cpu') and hasattr(self._array, 'numpy'):
            return self._array
        else:
            return torch.tensor(self._array)

    def plot(self, *args, **kwargs):
        """Plot the array using the geometry's plot method."""
        return self.geometry.plot(self._array, *args, **kwargs)

    def __len__(self):
        """Return the length of the array, handling scalar values."""
        try:
            return len(self._array)
        except TypeError:
            # Handle scalar values
            return 1

    def __getitem__(self, key):
        return self._array[key]

    def __setitem__(self, key, value):
        self._array[key] = value

    def __repr__(self):
        return f"CUQIarray({self._array}, is_par={self.is_par})"

    def __str__(self):
        return f"CUQIarray({self._array})"

    # Array methods
    def flatten(self):
        """Flatten the array."""
        return self._array.flatten()

    def reshape(self, *args, **kwargs):
        """Reshape the array."""
        return self._array.reshape(*args, **kwargs)

    def ravel(self):
        """Return a flattened array."""
        return self._array.ravel()

    def squeeze(self):
        """Remove single-dimensional entries from the shape of an array."""
        return self._array.squeeze()

    def transpose(self, *args, **kwargs):
        """Transpose the array."""
        return self._array.transpose(*args, **kwargs)

    @property
    def T(self):
        """Transpose of the array."""
        return self._array.T

    # Arithmetic operations
    def __add__(self, other):
        if isinstance(other, CUQIarray):
            return self._array + other._array
        else:
            return self._array + other

    def __sub__(self, other):
        if isinstance(other, CUQIarray):
            return self._array - other._array
        else:
            return self._array - other

    def __mul__(self, other):
        if isinstance(other, CUQIarray):
            return self._array * other._array
        else:
            return self._array * other

    def __rmul__(self, other):
        return other * self._array

    def __truediv__(self, other):
        if isinstance(other, CUQIarray):
            return self._array / other._array
        else:
            return self._array / other

    def __rtruediv__(self, other):
        return other / self._array

    def __matmul__(self, other):
        if isinstance(other, CUQIarray):
            return self._array @ other._array
        else:
            return self._array @ other

    def __rmatmul__(self, other):
        if isinstance(other, CUQIarray):
            return other._array @ self._array
        else:
            return other @ self._array

    def __pow__(self, other):
        return self._array ** other

    # Comparison operators
    def __lt__(self, other):
        if isinstance(other, CUQIarray):
            return self._array < other._array
        else:
            return self._array < other

    def __le__(self, other):
        if isinstance(other, CUQIarray):
            return self._array <= other._array
        else:
            return self._array <= other

    def __eq__(self, other):
        if isinstance(other, CUQIarray):
            return self._array == other._array
        else:
            return self._array == other

    def __ne__(self, other):
        if isinstance(other, CUQIarray):
            return self._array != other._array
        else:
            return self._array != other

    def __gt__(self, other):
        if isinstance(other, CUQIarray):
            return self._array > other._array
        else:
            return self._array > other

    def __ge__(self, other):
        if isinstance(other, CUQIarray):
            return self._array >= other._array
        else:
            return self._array >= other

    # Absolute value support
    def __abs__(self):
        return abs(self._array)
