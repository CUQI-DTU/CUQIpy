Array Backends
==============

CUQIpy provides an array-agnostic framework that allows you to seamlessly switch between different array backends for improved performance, GPU acceleration, or automatic differentiation capabilities.

Overview
--------

The array backend system abstracts all array operations through a unified interface ``cuqi.array``, allowing the same code to work with different underlying array libraries:

* **NumPy** (default): Standard CPU-based arrays
* **PyTorch**: GPU acceleration and automatic differentiation
* **CuPy**: GPU-accelerated arrays (CUDA)
* **JAX**: JIT compilation and automatic differentiation

Usage
-----

Basic Usage
~~~~~~~~~~~

Import the array backend and use it like NumPy:

.. code-block:: python

    import cuqi.array as xp
    
    # Create arrays
    x = xp.array([1, 2, 3])
    y = xp.zeros((3, 3))
    z = xp.dot(x, y)
    
    # Check current backend
    print(f"Current backend: {xp.get_backend_name()}")

Backend Selection
~~~~~~~~~~~~~~~~~

You can select the backend in several ways:

1. **Environment Variable** (recommended for global setting):

.. code-block:: bash

    export CUQI_ARRAY_BACKEND=pytorch
    python your_script.py

2. **Programmatic Selection** (recommended for runtime switching):

.. code-block:: python

    import cuqi.array as xp
    
    # Switch to PyTorch
    xp.set_backend("pytorch")
    
    # Switch back to NumPy
    xp.set_backend("numpy")

Supported Backends
------------------

NumPy Backend
~~~~~~~~~~~~~

The default backend providing standard CPU-based array operations:

.. code-block:: python

    xp.set_backend("numpy")
    x = xp.array([1.0, 2.0, 3.0])
    # Uses numpy.ndarray internally

**Features:**
- Full compatibility with all CUQIpy functions
- Mature and stable
- Extensive mathematical function support
- Default choice for most applications

PyTorch Backend
~~~~~~~~~~~~~~~

Provides GPU acceleration and automatic differentiation:

.. code-block:: python

    xp.set_backend("pytorch")
    
    # Basic arrays
    x = xp.array([1.0, 2.0, 3.0])  # torch.Tensor
    
    # Gradient tracking
    x = xp.array([2.0], requires_grad=True)
    y = x ** 2
    y.backward()
    print(x.grad)  # tensor([4.])

**Features:**
- GPU acceleration (when CUDA is available)
- Automatic differentiation for gradient-based inference
- Seamless integration with PyTorch ecosystem
- Ideal for gradient-based samplers and optimization

CuPy Backend
~~~~~~~~~~~~

GPU-accelerated arrays with NumPy-compatible API:

.. code-block:: python

    xp.set_backend("cupy")
    x = xp.array([1.0, 2.0, 3.0])  # cupy.ndarray on GPU

**Features:**
- GPU acceleration for large-scale computations
- NumPy-compatible API
- Efficient memory management on GPU

JAX Backend
~~~~~~~~~~~

JIT compilation and functional programming:

.. code-block:: python

    xp.set_backend("jax")
    x = xp.array([1.0, 2.0, 3.0])  # jax.numpy.ndarray

**Features:**
- JIT compilation for performance
- Functional programming paradigm
- Automatic differentiation
- Vectorization capabilities

Bayesian Inference with Backends
---------------------------------

The array backend system seamlessly integrates with CUQIpy's Bayesian inference capabilities:

Example: Backend Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import cuqi.array as xp
    from cuqi.distribution import GMRF, Gaussian
    from cuqi.model import LinearModel
    from cuqi.problem import BayesianProblem
    import numpy as np
    
    # Problem setup (using NumPy for initial data)
    n = 64
    A_data = np.random.randn(n, n) * 0.1 + np.eye(n)
    y_data = np.random.randn(n)
    
    # Test with NumPy backend
    xp.set_backend("numpy")
    A_numpy = LinearModel(xp.array(A_data))
    x = GMRF(mean=xp.zeros(n), prec=25.0, bc_type="zero")
    y = Gaussian(mean=A_numpy@x, cov=0.01)
    BP_numpy = BayesianProblem(y, x)
    BP_numpy.set_data(y=xp.array(y_data))
    x_map_numpy = BP_numpy.MAP()
    
    # Test with PyTorch backend
    xp.set_backend("pytorch")
    A_torch = LinearModel(xp.array(A_data))
    x = GMRF(mean=xp.zeros(n), prec=25.0, bc_type="zero")
    y = Gaussian(mean=A_torch@x, cov=0.01)
    BP_torch = BayesianProblem(y, x)
    BP_torch.set_data(y=xp.array(y_data))
    
    # Results should be equivalent (within numerical tolerance)
    print(f"NumPy backend: {xp.get_backend_name()}")

Automatic Differentiation
~~~~~~~~~~~~~~~~~~~~~~~~~~

PyTorch backend enables automatic differentiation for gradient-based methods:

.. code-block:: python

    xp.set_backend("pytorch")
    
    # Create parameter with gradient tracking
    x = xp.array([1.0, 2.0], requires_grad=True)
    
    # Define log-posterior (simplified example)
    log_post = -0.5 * xp.sum(x**2)  # Gaussian prior
    
    # Compute gradients
    log_post.backward()
    print(f"Gradient: {x.grad}")  # [-1., -2.]

Performance Considerations
--------------------------

Backend Selection Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Choose your backend based on your specific needs:

**NumPy**: 
- Small to medium problems (< 10,000 parameters)
- CPU-only environments
- Maximum compatibility
- Development and prototyping

**PyTorch**:
- Gradient-based inference methods (HMC, NUTS, MALA)
- GPU-accelerated computations
- Large-scale problems
- Automatic differentiation requirements

**CuPy**:
- Large-scale linear algebra operations
- GPU acceleration without gradient requirements
- NumPy code migration to GPU

**JAX**:
- Performance-critical applications
- Functional programming preference
- Advanced vectorization needs

Memory Management
~~~~~~~~~~~~~~~~~

Different backends have different memory management strategies:

.. code-block:: python

    # Convert between backends when needed
    xp.set_backend("pytorch")
    x_torch = xp.array([1, 2, 3])
    
    # Convert to NumPy for compatibility
    x_numpy = xp.to_numpy(x_torch)
    
    # Convert back
    xp.set_backend("numpy")
    x_np = xp.array(x_numpy)

Best Practices
--------------

1. **Start with NumPy**: Begin development with the NumPy backend for maximum compatibility.

2. **Switch for Performance**: Move to specialized backends (PyTorch, CuPy) when you need specific features.

3. **Test Consistency**: Always verify that results are consistent across backends:

   .. code-block:: python

       # Test numerical consistency
       xp.set_backend("numpy")
       result_numpy = your_computation()
       
       xp.set_backend("pytorch") 
       result_torch = your_computation()
       
       assert np.allclose(result_numpy, xp.to_numpy(result_torch))

4. **Use Environment Variables**: Set ``CUQI_ARRAY_BACKEND`` for consistent backend selection across runs.

5. **Handle Backend-Specific Features**: Some functions may not be available in all backends:

   .. code-block:: python

       try:
           result = xp.some_function(data)
       except NotImplementedError:
           print(f"Function not available in {xp.get_backend_name()} backend")

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**ImportError**: Backend not installed
  Install the required backend: ``pip install torch`` for PyTorch

**NotImplementedError**: Function not available
  Some specialized functions may not be implemented for all backends

**dtype Mismatches**: Different backends may have different default dtypes
  Explicitly specify ``dtype=xp.float64`` for consistency

**Memory Issues**: GPU backends may have memory limitations
  Monitor GPU memory usage and batch computations accordingly

Getting Help
~~~~~~~~~~~~

If you encounter issues with the array backend system:

1. Check the current backend: ``xp.get_backend_name()``
2. Verify backend installation: ``import torch`` (for PyTorch)
3. Test with NumPy backend first to isolate backend-specific issues
4. Report issues on the CUQIpy GitHub repository with backend information

Future Developments
-------------------

The array backend system is actively developed. Planned features include:

- Additional backend support (TensorFlow, Dask)
- Improved automatic backend selection
- Enhanced performance optimizations
- Better error handling and diagnostics

Stay updated with the latest developments by following the CUQIpy documentation and GitHub repository.