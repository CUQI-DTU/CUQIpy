# CUQIpy Array Backends

CUQIpy now supports multiple array backends, allowing you to choose between NumPy, CuPy, JAX, and potentially other array libraries. This provides flexibility for different computational needs, including GPU acceleration and just-in-time compilation.

## Supported Backends

### NumPy (Default)
- **Backend name**: `numpy`
- **Description**: Standard NumPy arrays for CPU computation
- **Installation**: Automatically available
- **Use case**: General purpose, CPU-based computations

### CuPy
- **Backend name**: `cupy`
- **Description**: GPU-accelerated arrays using CUDA
- **Installation**: `pip install cupy`
- **Use case**: GPU acceleration for NVIDIA GPUs

### PyTorch
- **Backend name**: `pytorch` or `torch`
- **Description**: Deep learning framework with GPU support and automatic differentiation
- **Installation**: `pip install torch`
- **Use case**: Deep learning, GPU acceleration, automatic differentiation

### JAX
- **Backend name**: `jax`
- **Description**: Arrays with just-in-time compilation and automatic differentiation
- **Installation**: `pip install jax`
- **Use case**: High-performance computing, automatic differentiation

## Backend Selection

### Environment Variable (Recommended)
Set the `CUQI_ARRAY_BACKEND` environment variable before importing CUQIpy:

```bash
# Use NumPy backend (default)
export CUQI_ARRAY_BACKEND=numpy

# Use CuPy backend for GPU acceleration
export CUQI_ARRAY_BACKEND=cupy

# Use JAX backend for JIT compilation
export CUQI_ARRAY_BACKEND=jax
```

### Programmatic Selection
You can also set the backend programmatically:

```python
import cuqi.array as xp

# Switch to CuPy backend
xp.set_backend('cupy')

# Check current backend
print(f"Current backend: {xp.get_backend_name()}")
```

## Usage Examples

### Basic Array Operations
```python
import cuqi.array as xp

# Create arrays
x = xp.array([1, 2, 3, 4])
y = xp.zeros(4)
z = xp.ones(4)

# Mathematical operations
result = x + z
dot_product = xp.dot(x, z)
sum_result = xp.sum(x)

print(f"Backend: {xp.get_backend_name()}")
print(f"Result: {result}")
print(f"Dot product: {dot_product}")
print(f"Sum: {sum_result}")
```

### Using with CUQIpy Distributions
```python
import cuqi.array as xp
from cuqi.distribution import Gaussian
from cuqi.geometry import Discrete

# Create a Gaussian distribution
mean = xp.zeros(5)
cov = xp.eye(5)
dist = Gaussian(mean=mean, cov=cov, geometry=Discrete(5))

# Sample from the distribution
samples = dist.sample(100)
print(f"Samples shape: {samples.shape}")
```

### Backend Conversion
```python
import cuqi.array as xp

# Create array with current backend
x = xp.array([1, 2, 3, 4])

# Convert to NumPy (regardless of current backend)
numpy_array = xp.to_numpy(x)

# Convert from NumPy to current backend
backend_array = xp.from_numpy(numpy_array)
```

## Backend Compatibility

### Automatic Fallback
If a requested backend is not available, CUQIpy will automatically fall back to NumPy with a warning:

```python
# If CuPy is not installed
import os
os.environ['CUQI_ARRAY_BACKEND'] = 'cupy'
import cuqi.array as xp  # Will show warning and use NumPy
```

### Function Coverage
The backend abstraction covers most commonly used array operations:

- **Array creation**: `array`, `zeros`, `ones`, `eye`, `linspace`, etc.
- **Mathematical operations**: `sin`, `cos`, `exp`, `log`, `sqrt`, etc.
- **Linear algebra**: `dot`, `matmul`, `linalg.solve`, `linalg.inv`, etc.
- **Array manipulation**: `reshape`, `transpose`, `concatenate`, etc.
- **Random numbers**: `random.normal`, `random.rand`, etc.

### Backend-Specific Features
Some functions may not be available on all backends. In such cases, the system will either:
1. Provide a fallback implementation
2. Use NumPy for that specific operation
3. Raise a `NotImplementedError` with a clear message

## Performance Considerations

### NumPy Backend
- Best for: Small to medium-sized problems, prototyping
- Memory: CPU RAM
- Parallelization: Limited (depends on BLAS library)

### CuPy Backend
- Best for: Large-scale problems, matrix operations
- Memory: GPU VRAM (limited by GPU memory)
- Parallelization: Excellent (thousands of CUDA cores)
- Note: Data transfer between CPU and GPU can be a bottleneck

### JAX Backend
- Best for: Iterative algorithms, gradient-based optimization
- Memory: CPU RAM or GPU VRAM (depending on device)
- Parallelization: Excellent with JIT compilation
- Note: First run includes compilation overhead

## Migration Guide

### From NumPy to Backend-Agnostic Code

**Before:**
```python
import numpy as np

x = np.array([1, 2, 3])
y = np.zeros(3)
result = np.dot(x, y)
```

**After:**
```python
import cuqi.array as xp

x = xp.array([1, 2, 3])
y = xp.zeros(3)
result = xp.dot(x, y)
```

### Updating Existing Code
1. Replace `import numpy as np` with `import cuqi.array as xp`
2. Replace `np.function()` with `xp.function()`
3. Test with different backends to ensure compatibility

## Troubleshooting

### Common Issues

1. **Backend not found**: Install the required backend package
2. **Memory errors with GPU**: Reduce problem size or use CPU backend
3. **Slow performance**: Check if you're using the appropriate backend for your hardware

### Debugging
```python
import cuqi.array as xp

# Check current backend
print(f"Backend: {xp.get_backend_name()}")

# Test basic operations
x = xp.array([1, 2, 3])
print(f"Array type: {type(x)}")
print(f"Array: {x}")
```

### Getting Help
If you encounter issues with the backend system:
1. Check that the backend is properly installed
2. Verify environment variable settings
3. Test with the NumPy backend first
4. Report issues on the CUQIpy GitHub repository

## Advanced Usage

### Custom Backend Extensions
For advanced users, the backend system can be extended to support additional array libraries by modifying the `cuqi/array/__init__.py` file.

### Performance Profiling
```python
import time
import cuqi.array as xp

# Time array operations
x = xp.random.randn(1000, 1000)
y = xp.random.randn(1000, 1000)

start = time.time()
result = xp.dot(x, y)
end = time.time()

print(f"Backend: {xp.get_backend_name()}")
print(f"Time: {end - start:.4f} seconds")
```

This backend system provides a solid foundation for array-agnostic computing in CUQIpy while maintaining backward compatibility and ease of use.