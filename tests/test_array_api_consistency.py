"""
Test array API consistency between NumPy and PyTorch backends.

This module tests that all array operations produce consistent results
between different backends, ensuring the array-agnostic interface works correctly.
"""

import pytest
import numpy as np
import cuqi.array as xp


class TestArrayAPIConsistency:
    """Test consistency of array API methods across backends."""
    
    @pytest.fixture(autouse=True)
    def setup_backends(self):
        """Setup test to ensure we have both backends available."""
        # Store original backend
        self.original_backend = xp.get_backend_name()
        
        # Check if PyTorch is available
        try:
            xp.set_backend("pytorch")
            self.pytorch_available = True
        except ImportError:
            self.pytorch_available = False
        
        # Reset to original backend
        xp.set_backend(self.original_backend)
        
        yield
        
        # Cleanup: restore original backend
        xp.set_backend(self.original_backend)
    
    def compare_backends(self, func_name, *args, **kwargs):
        """Compare function results between NumPy and PyTorch backends."""
        if not self.pytorch_available:
            pytest.skip("PyTorch not available")
        
        # Test with NumPy backend
        xp.set_backend("numpy")
        numpy_func = getattr(xp, func_name)
        numpy_result = numpy_func(*args, **kwargs)
        
        # Test with PyTorch backend
        xp.set_backend("pytorch")
        pytorch_func = getattr(xp, func_name)
        
        # Convert args to PyTorch backend arrays if they are arrays (but not shapes)
        pytorch_args = []
        for i, arg in enumerate(args):
            # Don't convert tuples that are likely to be shapes (first arg for zeros, ones, etc.)
            if isinstance(arg, tuple) and all(isinstance(x, (int, np.integer)) for x in arg):
                pytorch_args.append(arg)
            elif isinstance(arg, list) and len(arg) > 0:
                # Check if this is a list of arrays (for functions like vstack, hstack)
                # vs a single array represented as a list
                if (func_name in ['vstack', 'hstack', 'dstack', 'concatenate', 'stack'] and 
                    (hasattr(arg[0], '__array__') or isinstance(arg[0], (list, np.ndarray)))):
                    # List of arrays - convert each element
                    pytorch_arrays = [xp.array(a) for a in arg]
                    pytorch_args.append(pytorch_arrays)
                else:
                    # Single array represented as list - convert the whole thing
                    pytorch_args.append(xp.array(arg))
            elif hasattr(arg, '__array__') or isinstance(arg, np.ndarray):
                try:
                    pytorch_args.append(xp.array(arg))
                except:
                    pytorch_args.append(arg)
            else:
                pytorch_args.append(arg)
        
        pytorch_result = pytorch_func(*pytorch_args, **kwargs)
        
        # Convert results back to numpy for comparison
        if hasattr(numpy_result, '__array__'):
            numpy_final = np.asarray(numpy_result)
        else:
            numpy_final = numpy_result
            
        if hasattr(pytorch_result, '__array__') or hasattr(pytorch_result, 'detach'):
            pytorch_final = xp.to_numpy(pytorch_result)
        else:
            pytorch_final = pytorch_result
        
        return numpy_final, pytorch_final
    
    def assert_results_close(self, numpy_result, pytorch_result, rtol=1e-7, atol=1e-8):
        """Assert that results from different backends are close."""
        if isinstance(numpy_result, (int, float, complex, np.number)):
            if isinstance(pytorch_result, (int, float, complex, np.number)):
                np.testing.assert_allclose(numpy_result, pytorch_result, rtol=rtol, atol=atol)
            else:
                pytest.fail(f"Type mismatch: NumPy result is scalar, PyTorch result is {type(pytorch_result)}")
        elif hasattr(numpy_result, 'shape'):
            assert numpy_result.shape == pytorch_result.shape, f"Shape mismatch: {numpy_result.shape} vs {pytorch_result.shape}"
            np.testing.assert_allclose(numpy_result, pytorch_result, rtol=rtol, atol=atol)
        else:
            assert numpy_result == pytorch_result, f"Results don't match: {numpy_result} vs {pytorch_result}"

    # Array creation functions
    @pytest.mark.parametrize("data,dtype", [
        ([1, 2, 3], None),
        ([1.0, 2.0, 3.0], None),
        ([[1, 2], [3, 4]], None),
        ([1, 2, 3], np.float32),
        ([1, 2, 3], np.float64),
        ([1, 2, 3], np.int32),
    ])
    def test_array_creation(self, data, dtype):
        """Test array creation consistency."""
        numpy_result, pytorch_result = self.compare_backends("array", data, dtype=dtype)
        self.assert_results_close(numpy_result, pytorch_result)
        
        # Check dtype consistency
        if dtype is not None:
            assert numpy_result.dtype == dtype or np.issubdtype(numpy_result.dtype, dtype)

    @pytest.mark.parametrize("shape,dtype", [
        ((3,), None),
        ((2, 3), None),
        ((2, 3, 4), None),
        ((5,), np.float32),
        ((3, 3), np.float64),
    ])
    def test_zeros(self, shape, dtype):
        """Test zeros creation consistency."""
        numpy_result, pytorch_result = self.compare_backends("zeros", shape, dtype=dtype)
        self.assert_results_close(numpy_result, pytorch_result)

    @pytest.mark.parametrize("shape,dtype", [
        ((3,), None),
        ((2, 3), None),
        ((2, 3, 4), None),
        ((5,), np.float32),
        ((3, 3), np.float64),
    ])
    def test_ones(self, shape, dtype):
        """Test ones creation consistency."""
        numpy_result, pytorch_result = self.compare_backends("ones", shape, dtype=dtype)
        self.assert_results_close(numpy_result, pytorch_result)

    @pytest.mark.parametrize("n,dtype", [
        (3, None),
        (5, np.float32),
        (4, np.float64),
    ])
    def test_identity(self, n, dtype):
        """Test identity matrix creation consistency."""
        numpy_result, pytorch_result = self.compare_backends("identity", n, dtype=dtype)
        self.assert_results_close(numpy_result, pytorch_result)

    @pytest.mark.parametrize("n,k,dtype", [
        (3, 0, None),
        (4, 1, None),
        (3, -1, None),
        (5, 0, np.float32),
    ])
    def test_eye(self, n, k, dtype):
        """Test eye matrix creation consistency."""
        numpy_result, pytorch_result = self.compare_backends("eye", n, k=k, dtype=dtype)
        self.assert_results_close(numpy_result, pytorch_result)

    @pytest.mark.parametrize("start,stop,num", [
        (0, 10, 5),
        (0.0, 1.0, 11),
        (-1, 1, 21),
    ])
    def test_linspace(self, start, stop, num):
        """Test linspace consistency."""
        numpy_result, pytorch_result = self.compare_backends("linspace", start, stop, num)
        self.assert_results_close(numpy_result, pytorch_result)

    @pytest.mark.parametrize("start,stop,step", [
        (0, 10, 1),
        (0, 10, 2),
        (0.0, 5.0, 0.5),
        (-5, 5, 1),
    ])
    def test_arange(self, start, stop, step):
        """Test arange consistency."""
        numpy_result, pytorch_result = self.compare_backends("arange", start, stop, step)
        self.assert_results_close(numpy_result, pytorch_result)

    # Array manipulation functions
    @pytest.mark.parametrize("data,new_shape", [
        ([[1, 2, 3, 4]], (2, 2)),
        ([1, 2, 3, 4, 5, 6], (2, 3)),
        ([1, 2, 3, 4, 5, 6], (3, 2)),
        ([[1, 2], [3, 4]], (4,)),
    ])
    def test_reshape(self, data, new_shape):
        """Test reshape consistency."""
        numpy_result, pytorch_result = self.compare_backends("reshape", data, new_shape)
        self.assert_results_close(numpy_result, pytorch_result)

    @pytest.mark.parametrize("data,axis", [
        ([[1, 2], [3, 4]], None),
        ([[1, 2], [3, 4]], 0),
        ([[1, 2], [3, 4]], 1),
        ([[[1, 2]], [[3, 4]]], 2),
    ])
    def test_squeeze(self, data, axis):
        """Test squeeze consistency."""
        # Add dimensions that can be squeezed
        data_with_dims = np.array(data)
        if axis is None:
            data_with_dims = data_with_dims[..., np.newaxis]
        
        numpy_result, pytorch_result = self.compare_backends("squeeze", data_with_dims, axis=axis)
        self.assert_results_close(numpy_result, pytorch_result)

    @pytest.mark.parametrize("data,axis", [
        ([1, 2, 3], 0),
        ([1, 2, 3], 1),
        ([[1, 2], [3, 4]], 0),
        ([[1, 2], [3, 4]], 2),
    ])
    def test_expand_dims(self, data, axis):
        """Test expand_dims consistency."""
        numpy_result, pytorch_result = self.compare_backends("expand_dims", data, axis=axis)
        self.assert_results_close(numpy_result, pytorch_result)

    @pytest.mark.parametrize("data,axes", [
        ([[1, 2], [3, 4]], None),
        ([[1, 2], [3, 4]], (1, 0)),
        ([[[1, 2]], [[3, 4]]], (2, 0, 1)),
    ])
    def test_transpose(self, data, axes):
        """Test transpose consistency."""
        numpy_result, pytorch_result = self.compare_backends("transpose", data, axes=axes)
        self.assert_results_close(numpy_result, pytorch_result)

    # Stack and concatenation functions
    @pytest.mark.parametrize("arrays", [
        ([[1, 2], [3, 4]]),
        ([[1, 2, 3], [4, 5, 6]]),
        ([[[1, 2]], [[3, 4]]]),
    ])
    def test_vstack(self, arrays):
        """Test vstack consistency."""
        numpy_result, pytorch_result = self.compare_backends("vstack", arrays)
        self.assert_results_close(numpy_result, pytorch_result)

    @pytest.mark.parametrize("arrays", [
        ([[1, 2], [3, 4]]),
        ([[1], [2]]),
        ([[[1, 2]], [[3, 4]]]),
    ])
    def test_hstack(self, arrays):
        """Test hstack consistency."""
        numpy_result, pytorch_result = self.compare_backends("hstack", arrays)
        self.assert_results_close(numpy_result, pytorch_result)

    @pytest.mark.parametrize("arrays", [
        ([[1, 2], [3, 4]]),
        ([[[1, 2]], [[3, 4]]]),
    ])
    def test_dstack(self, arrays):
        """Test dstack consistency."""
        numpy_result, pytorch_result = self.compare_backends("dstack", arrays)
        self.assert_results_close(numpy_result, pytorch_result)

    @pytest.mark.parametrize("arrays,axis", [
        ([[1, 2], [3, 4]], 0),
        ([[1, 2], [3, 4]], 1),
        ([[[1, 2]], [[3, 4]]], 0),
    ])
    def test_concatenate(self, arrays, axis):
        """Test concatenate consistency."""
        numpy_result, pytorch_result = self.compare_backends("concatenate", arrays, axis=axis)
        self.assert_results_close(numpy_result, pytorch_result)

    @pytest.mark.parametrize("arrays,axis", [
        ([[1, 2], [3, 4]], 0),
        ([[1, 2], [3, 4]], 1),
        ([[[1, 2]], [[3, 4]]], 2),
    ])
    def test_stack(self, arrays, axis):
        """Test stack consistency."""
        numpy_result, pytorch_result = self.compare_backends("stack", arrays, axis=axis)
        self.assert_results_close(numpy_result, pytorch_result)

    # Mathematical functions
    @pytest.mark.parametrize("data", [
        [1, 2, 3, 4],
        [[1, 2], [3, 4]],
        [0.1, 0.5, 1.0, 2.0],
        [-1, -2, 3, 4],
    ])
    def test_abs(self, data):
        """Test abs consistency."""
        numpy_result, pytorch_result = self.compare_backends("abs", data)
        self.assert_results_close(numpy_result, pytorch_result)

    @pytest.mark.parametrize("data", [
        [1, 2, 3, 4],
        [[1, 2], [3, 4]],
        [0.1, 0.5, 1.0, 2.0],
    ])
    def test_sqrt(self, data):
        """Test sqrt consistency."""
        numpy_result, pytorch_result = self.compare_backends("sqrt", data)
        self.assert_results_close(numpy_result, pytorch_result)

    @pytest.mark.parametrize("data", [
        [1, 2, 3, 4],
        [[1, 2], [3, 4]],
        [0.1, 0.5, 1.0, 2.0],
    ])
    def test_square(self, data):
        """Test square consistency."""
        numpy_result, pytorch_result = self.compare_backends("square", data)
        self.assert_results_close(numpy_result, pytorch_result)

    @pytest.mark.parametrize("data", [
        [0.1, 0.5, 1.0, 2.0],
        [[0.1, 0.5], [1.0, 2.0]],
        [1e-5, 1e-3, 1.0, 100.0],
    ])
    def test_log(self, data):
        """Test log consistency."""
        numpy_result, pytorch_result = self.compare_backends("log", data)
        self.assert_results_close(numpy_result, pytorch_result)

    @pytest.mark.parametrize("data", [
        [0.1, 0.5, 1.0, 2.0],
        [[0.1, 0.5], [1.0, 2.0]],
        [-2, -1, 0, 1, 2],
    ])
    def test_exp(self, data):
        """Test exp consistency."""
        numpy_result, pytorch_result = self.compare_backends("exp", data)
        self.assert_results_close(numpy_result, pytorch_result)

    @pytest.mark.parametrize("x1,x2", [
        ([2, 3, 4], [2, 2, 2]),
        ([[2, 3], [4, 5]], [[2, 2], [3, 3]]),
        ([2.0, 3.0], [0.5, 2.0]),
    ])
    def test_power(self, x1, x2):
        """Test power consistency."""
        numpy_result, pytorch_result = self.compare_backends("power", x1, x2)
        self.assert_results_close(numpy_result, pytorch_result)

    # Trigonometric functions
    @pytest.mark.parametrize("data", [
        [0, np.pi/4, np.pi/2, np.pi],
        [[0, np.pi/6], [np.pi/4, np.pi/3]],
        [-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
    ])
    def test_sin(self, data):
        """Test sin consistency."""
        numpy_result, pytorch_result = self.compare_backends("sin", data)
        self.assert_results_close(numpy_result, pytorch_result)

    @pytest.mark.parametrize("data", [
        [0, np.pi/4, np.pi/2, np.pi],
        [[0, np.pi/6], [np.pi/4, np.pi/3]],
        [-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
    ])
    def test_cos(self, data):
        """Test cos consistency."""
        numpy_result, pytorch_result = self.compare_backends("cos", data)
        self.assert_results_close(numpy_result, pytorch_result)

    @pytest.mark.parametrize("data", [
        [0, np.pi/4, np.pi/3],
        [[0, np.pi/6], [np.pi/4, np.pi/3]],
        [-np.pi/4, 0, np.pi/4],
    ])
    def test_tan(self, data):
        """Test tan consistency."""
        numpy_result, pytorch_result = self.compare_backends("tan", data)
        self.assert_results_close(numpy_result, pytorch_result)

    # Reduction functions
    @pytest.mark.parametrize("data,axis,keepdims", [
        ([[1, 2], [3, 4]], None, False),
        ([[1, 2], [3, 4]], 0, False),
        ([[1, 2], [3, 4]], 1, False),
        ([[1, 2], [3, 4]], 0, True),
        ([[[1, 2]], [[3, 4]]], 2, False),
    ])
    def test_sum(self, data, axis, keepdims):
        """Test sum consistency."""
        numpy_result, pytorch_result = self.compare_backends("sum", data, axis=axis, keepdims=keepdims)
        self.assert_results_close(numpy_result, pytorch_result)

    @pytest.mark.parametrize("data,axis,keepdims", [
        ([[1, 2], [3, 4]], None, False),
        ([[1, 2], [3, 4]], 0, False),
        ([[1, 2], [3, 4]], 1, False),
        ([[1, 2], [3, 4]], 0, True),
    ])
    def test_mean(self, data, axis, keepdims):
        """Test mean consistency."""
        numpy_result, pytorch_result = self.compare_backends("mean", data, axis=axis, keepdims=keepdims)
        self.assert_results_close(numpy_result, pytorch_result)

    @pytest.mark.parametrize("data,axis,keepdims", [
        ([[1, 2], [3, 4]], None, False),
        ([[1, 2], [3, 4]], 0, False),
        ([[1, 2], [3, 4]], 1, False),
        ([[1, 2], [3, 4]], 0, True),
    ])
    def test_max(self, data, axis, keepdims):
        """Test max consistency."""
        numpy_result, pytorch_result = self.compare_backends("max", data, axis=axis, keepdims=keepdims)
        self.assert_results_close(numpy_result, pytorch_result)

    @pytest.mark.parametrize("data,axis,keepdims", [
        ([[1, 2], [3, 4]], None, False),
        ([[1, 2], [3, 4]], 0, False),
        ([[1, 2], [3, 4]], 1, False),
        ([[1, 2], [3, 4]], 0, True),
    ])
    def test_min(self, data, axis, keepdims):
        """Test min consistency."""
        numpy_result, pytorch_result = self.compare_backends("min", data, axis=axis, keepdims=keepdims)
        self.assert_results_close(numpy_result, pytorch_result)

    # Linear algebra functions
    @pytest.mark.parametrize("a,b", [
        ([1, 2, 3], [4, 5, 6]),
        ([[1, 2], [3, 4]], [[5, 6], [7, 8]]),
        ([[1, 2, 3]], [[4], [5], [6]]),
    ])
    def test_dot(self, a, b):
        """Test dot product consistency."""
        numpy_result, pytorch_result = self.compare_backends("dot", a, b)
        self.assert_results_close(numpy_result, pytorch_result)

    @pytest.mark.parametrize("a,b", [
        ([[1, 2], [3, 4]], [[5, 6], [7, 8]]),
        ([[1, 2, 3]], [[4], [5], [6]]),
        ([[1], [2]], [[3, 4, 5]]),
    ])
    def test_matmul(self, a, b):
        """Test matrix multiplication consistency."""
        numpy_result, pytorch_result = self.compare_backends("matmul", a, b)
        self.assert_results_close(numpy_result, pytorch_result)

    # Comparison and logical functions
    @pytest.mark.parametrize("a,b", [
        ([1, 2, 3], [1, 3, 2]),
        ([[1, 2], [3, 4]], [[1, 3], [2, 4]]),
        ([1.0, 2.0, 3.0], [1.0, 2.1, 2.9]),
    ])
    def test_equal(self, a, b):
        """Test equal comparison consistency."""
        numpy_result, pytorch_result = self.compare_backends("equal", a, b)
        self.assert_results_close(numpy_result, pytorch_result)

    @pytest.mark.parametrize("a,b", [
        ([1, 2, 3], [1, 3, 2]),
        ([[1, 2], [3, 4]], [[1, 3], [2, 4]]),
        ([1.0, 2.0, 3.0], [1.0, 2.1, 2.9]),
    ])
    def test_greater(self, a, b):
        """Test greater comparison consistency."""
        numpy_result, pytorch_result = self.compare_backends("greater", a, b)
        self.assert_results_close(numpy_result, pytorch_result)

    @pytest.mark.parametrize("a,b", [
        ([1, 2, 3], [1, 3, 2]),
        ([[1, 2], [3, 4]], [[1, 3], [2, 4]]),
        ([1.0, 2.0, 3.0], [1.0, 2.1, 2.9]),
    ])
    def test_less(self, a, b):
        """Test less comparison consistency."""
        numpy_result, pytorch_result = self.compare_backends("less", a, b)
        self.assert_results_close(numpy_result, pytorch_result)

    # Utility functions
    @pytest.mark.parametrize("element", [
        5,
        5.0,
        [5],
        [[5]],
        np.array(5),
        np.array([5]),
    ])
    def test_isscalar(self, element):
        """Test isscalar consistency."""
        numpy_result, pytorch_result = self.compare_backends("isscalar", element)
        assert numpy_result == pytorch_result

    @pytest.mark.parametrize("data", [
        [1, 2, 3, 4],
        [[1, 2], [3, 4]],
        [[[1, 2]], [[3, 4]]],
    ])
    def test_size(self, data):
        """Test size consistency."""
        numpy_result, pytorch_result = self.compare_backends("size", data)
        assert numpy_result == pytorch_result

    @pytest.mark.parametrize("data", [
        [1, 2, 3, 4],
        [[1, 2], [3, 4]],
        [[[1, 2]], [[3, 4]]],
    ])
    def test_shape(self, data):
        """Test shape consistency."""
        numpy_result, pytorch_result = self.compare_backends("shape", data)
        assert numpy_result == tuple(pytorch_result) or tuple(numpy_result) == pytorch_result

    @pytest.mark.parametrize("data", [
        [1, 2, 3, 4],
        [[1, 2], [3, 4]],
        [[[1, 2]], [[3, 4]]],
    ])
    def test_ndim(self, data):
        """Test ndim consistency."""
        numpy_result, pytorch_result = self.compare_backends("ndim", data)
        assert numpy_result == pytorch_result

    # Array equality functions
    @pytest.mark.parametrize("a,b", [
        ([1, 2, 3], [1, 2, 3]),
        ([1, 2, 3], [1, 2, 4]),
        ([[1, 2], [3, 4]], [[1, 2], [3, 4]]),
        ([[1, 2], [3, 4]], [[1, 2], [3, 5]]),
    ])
    def test_array_equal(self, a, b):
        """Test array_equal consistency."""
        numpy_result, pytorch_result = self.compare_backends("array_equal", a, b)
        assert numpy_result == pytorch_result

    @pytest.mark.parametrize("a,b", [
        ([1, 2, 3], [1, 2, 3]),
        ([1, 2, 3], [1, 2, 4]),
        ([[1, 2], [3, 4]], [[1, 2], [3, 4]]),
        ([1, 2, 3], [[1], [2], [3]]),  # Test broadcasting
    ])
    def test_array_equiv(self, a, b):
        """Test array_equiv consistency."""
        numpy_result, pytorch_result = self.compare_backends("array_equiv", a, b)
        assert numpy_result == pytorch_result

    @pytest.mark.parametrize("a,b,rtol,atol", [
        ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], 1e-5, 1e-8),
        ([1.0, 2.0, 3.0], [1.0, 2.00001, 3.0], 1e-4, 1e-8),
        ([[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]], 1e-5, 1e-8),
    ])
    def test_allclose(self, a, b, rtol, atol):
        """Test allclose consistency."""
        numpy_result, pytorch_result = self.compare_backends("allclose", a, b, rtol=rtol, atol=atol)
        assert numpy_result == pytorch_result


class TestSparseMatrixConsistency:
    """Test consistency of sparse matrix functions across backends."""
    
    @pytest.fixture(autouse=True)
    def setup_backends(self):
        """Setup test to ensure we have both backends available."""
        # Store original backend
        self.original_backend = xp.get_backend_name()
        
        # Check if PyTorch is available
        try:
            xp.set_backend("pytorch")
            self.pytorch_available = True
        except ImportError:
            self.pytorch_available = False
        
        # Reset to original backend
        xp.set_backend(self.original_backend)
        
        yield
        
        # Cleanup: restore original backend
        xp.set_backend(self.original_backend)
    
    def test_sparse_spdiags_consistency(self):
        """Test sparse_spdiags consistency between backends."""
        if not self.pytorch_available:
            pytest.skip("PyTorch not available")
        
        # Test data
        data = np.array([[-1, -1, -1, -1], [1, 1, 1, 1]])
        diags = [-1, 0]
        m, n = 5, 4
        
        # NumPy backend
        xp.set_backend("numpy")
        numpy_result = xp.sparse_spdiags(data, diags, m, n)
        
        # PyTorch backend
        xp.set_backend("pytorch")
        pytorch_data = xp.array(data)
        pytorch_result = xp.sparse_spdiags(pytorch_data, diags, m, n)
        
        # Compare results (both should be scipy sparse matrices)
        assert numpy_result.shape == pytorch_result.shape
        np.testing.assert_array_equal(numpy_result.toarray(), pytorch_result.toarray())
    
    def test_sparse_eye_consistency(self):
        """Test sparse_eye consistency between backends."""
        if not self.pytorch_available:
            pytest.skip("PyTorch not available")
        
        n = 5
        
        # NumPy backend
        xp.set_backend("numpy")
        numpy_result = xp.sparse_eye(n)
        
        # PyTorch backend
        xp.set_backend("pytorch")
        pytorch_result = xp.sparse_eye(n)
        
        # Compare results
        assert numpy_result.shape == pytorch_result.shape
        np.testing.assert_array_equal(numpy_result.toarray(), pytorch_result.toarray())