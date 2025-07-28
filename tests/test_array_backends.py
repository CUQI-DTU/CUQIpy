"""
Tests for array backend consistency and functionality.

This module tests that different array backends (NumPy, PyTorch) produce
consistent results for basic operations and Bayesian inference tasks.
"""

import pytest
import numpy as np
import cuqi.array as xp
from cuqi.distribution import GMRF, Gaussian
from cuqi.model import LinearModel
from cuqi.problem import BayesianProblem


class TestArrayBackends:
    """Test array backend functionality and consistency."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Always start with numpy backend
        xp.set_backend("numpy")
        
        # Create test data
        self.n = 16  # Small size for fast tests
        np.random.seed(42)  # For reproducibility
        self.A_data = np.random.randn(self.n, self.n) * 0.1 + np.eye(self.n)
        self.x_true = np.random.randn(self.n)
        self.y_data = self.A_data @ self.x_true + 0.01 * np.random.randn(self.n)
    
    def teardown_method(self):
        """Clean up after tests."""
        # Reset to numpy backend
        xp.set_backend("numpy")
    
    def test_backend_switching(self):
        """Test that backend switching works correctly."""
        # Test numpy backend
        xp.set_backend("numpy")
        assert xp.get_backend_name() == "numpy"
        
        # Test pytorch backend (if available)
        try:
            xp.set_backend("pytorch")
            assert xp.get_backend_name() == "pytorch"
        except ImportError:
            pytest.skip("PyTorch not available")
    
    def test_basic_array_operations_consistency(self):
        """Test that basic array operations are consistent across backends."""
        # Test data
        x_data = [1.0, 2.0, 3.0]
        y_data = [4.0, 5.0, 6.0]
        
        # Test with NumPy
        xp.set_backend("numpy")
        x_np = xp.array(x_data, dtype=xp.float64)
        y_np = xp.array(y_data, dtype=xp.float64)
        
        # Basic operations
        sum_np = x_np + y_np
        dot_np = xp.dot(x_np, y_np)
        norm_np = xp.linalg.norm(x_np)
        
        # Test with PyTorch (if available)
        try:
            xp.set_backend("pytorch")
            x_torch = xp.array(x_data, dtype=xp.float64)
            y_torch = xp.array(y_data, dtype=xp.float64)
            
            # Basic operations
            sum_torch = x_torch + y_torch
            dot_torch = xp.dot(x_torch, y_torch)
            norm_torch = xp.linalg.norm(x_torch)
            
            # Convert to numpy for comparison
            sum_torch_np = xp.to_numpy(sum_torch)
            dot_torch_np = xp.to_numpy(dot_torch)
            norm_torch_np = xp.to_numpy(norm_torch)
            
            # Check consistency
            np.testing.assert_allclose(sum_np, sum_torch_np, rtol=1e-10)
            np.testing.assert_allclose(dot_np, dot_torch_np, rtol=1e-10)
            np.testing.assert_allclose(norm_np, norm_torch_np, rtol=1e-10)
            
        except ImportError:
            pytest.skip("PyTorch not available")
    
    def test_gradient_computation_pytorch(self):
        """Test PyTorch gradient computation functionality."""
        try:
            xp.set_backend("pytorch")
        except ImportError:
            pytest.skip("PyTorch not available")
        
        # Create tensor with gradient tracking
        x = xp.array([2.0], requires_grad=True, dtype=xp.float64)
        
        # Simple quadratic function: f(x) = x^2
        y = x ** 2
        
        # Compute gradient
        y.backward()
        
        # Check gradient (should be 2*x = 4.0)
        expected_grad = 4.0
        actual_grad = x.grad.item()
        
        assert abs(actual_grad - expected_grad) < 1e-10
    
    def test_posterior_logpdf_consistency(self):
        """Test that posterior logpdf values are consistent across backends."""
        # Test point
        x_test = np.random.randn(self.n)
        
        # NumPy backend
        xp.set_backend("numpy")
        A_np = LinearModel(xp.array(self.A_data, dtype=xp.float64))
        x_prior_np = GMRF(mean=xp.zeros(self.n, dtype=xp.float64), prec=1.0, bc_type="zero")
        y_like_np = Gaussian(mean=A_np@x_prior_np, cov=0.01)
        BP_np = BayesianProblem(y_like_np, x_prior_np)
        BP_np.set_data(y=xp.array(self.y_data, dtype=xp.float64))
        
        logpdf_np = BP_np.posterior.logpdf(xp.array(x_test, dtype=xp.float64))
        
        # PyTorch backend
        try:
            xp.set_backend("pytorch")
            A_torch = LinearModel(xp.array(self.A_data, dtype=xp.float64))
            x_prior_torch = GMRF(mean=xp.zeros(self.n, dtype=xp.float64), prec=1.0, bc_type="zero")
            y_like_torch = Gaussian(mean=A_torch@x_prior_torch, cov=0.01)
            BP_torch = BayesianProblem(y_like_torch, x_prior_torch)
            BP_torch.set_data(y=xp.array(self.y_data, dtype=xp.float64))
            
            logpdf_torch = BP_torch.posterior.logpdf(xp.array(x_test, dtype=xp.float64))
            logpdf_torch_np = xp.to_numpy(logpdf_torch)
            
            # Check consistency (allow for small numerical differences)
            np.testing.assert_allclose(logpdf_np, logpdf_torch_np, rtol=1e-6, atol=1e-8)
            
        except ImportError:
            pytest.skip("PyTorch not available")
    
    def test_posterior_gradient_pytorch(self):
        """Test that posterior gradients can be computed with PyTorch."""
        try:
            xp.set_backend("pytorch")
        except ImportError:
            pytest.skip("PyTorch not available")
        
        # Set up problem
        A_torch = LinearModel(xp.array(self.A_data, dtype=xp.float64))
        x_prior = GMRF(mean=xp.zeros(self.n, dtype=xp.float64), prec=1.0, bc_type="zero")
        y_like = Gaussian(mean=A_torch@x_prior, cov=0.01)
        BP = BayesianProblem(y_like, x_prior)
        BP.set_data(y=xp.array(self.y_data, dtype=xp.float64))
        
        # Test point with gradient tracking
        x_test = xp.array(np.random.randn(self.n), requires_grad=True, dtype=xp.float64)
        
        # Compute log posterior
        logpdf = BP.posterior.logpdf(x_test)
        
        # Compute gradient
        logpdf.backward()
        
        # Check that gradient was computed
        assert x_test.grad is not None
        assert x_test.grad.shape == (self.n,)
        
        # Check that gradient has reasonable magnitude
        grad_norm = xp.linalg.norm(x_test.grad).item()
        assert grad_norm > 0  # Should be non-zero
        assert grad_norm < 1000  # Should be reasonable magnitude
    
    def test_array_dtype_consistency(self):
        """Test that dtypes are handled consistently across backends."""
        test_data = [1.0, 2.0, 3.0]
        
        # NumPy backend
        xp.set_backend("numpy")
        x_np = xp.array(test_data, dtype=xp.float64)
        assert x_np.dtype == xp.float64
        
        # PyTorch backend
        try:
            xp.set_backend("pytorch")
            x_torch = xp.array(test_data, dtype=xp.float64)
            assert x_torch.dtype == xp.float64
            
            # Check conversion consistency
            x_converted = xp.to_numpy(x_torch)
            np.testing.assert_allclose(x_np, x_converted, rtol=1e-15)
            
        except ImportError:
            pytest.skip("PyTorch not available")
    
    def test_mathematical_functions_consistency(self):
        """Test that mathematical functions are consistent across backends."""
        # Test data
        x_data = np.array([1.0, 2.0, 3.0, 4.0])
        
        # NumPy backend
        xp.set_backend("numpy")
        x_np = xp.array(x_data, dtype=xp.float64)
        
        # Mathematical operations
        sin_np = xp.sin(x_np)
        exp_np = xp.exp(x_np)
        sqrt_np = xp.sqrt(x_np)
        sum_np = xp.sum(x_np)
        
        # PyTorch backend
        try:
            xp.set_backend("pytorch")
            x_torch = xp.array(x_data, dtype=xp.float64)
            
            # Mathematical operations
            sin_torch = xp.sin(x_torch)
            exp_torch = xp.exp(x_torch)
            sqrt_torch = xp.sqrt(x_torch)
            sum_torch = xp.sum(x_torch)
            
            # Convert to numpy for comparison
            sin_torch_np = xp.to_numpy(sin_torch)
            exp_torch_np = xp.to_numpy(exp_torch)
            sqrt_torch_np = xp.to_numpy(sqrt_torch)
            sum_torch_np = xp.to_numpy(sum_torch)
            
            # Check consistency
            np.testing.assert_allclose(sin_np, sin_torch_np, rtol=1e-10)
            np.testing.assert_allclose(exp_np, exp_torch_np, rtol=1e-10)
            np.testing.assert_allclose(sqrt_np, sqrt_torch_np, rtol=1e-10)
            np.testing.assert_allclose(sum_np, sum_torch_np, rtol=1e-10)
            
        except ImportError:
            pytest.skip("PyTorch not available")


class TestBackendSpecificFunctions:
    """Test backend-specific function implementations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        xp.set_backend("numpy")
    
    def teardown_method(self):
        """Clean up after tests."""
        xp.set_backend("numpy")
    
    def test_piecewise_function(self):
        """Test piecewise function implementation."""
        x = xp.array([0.5, 1.5, 2.5], dtype=xp.float64)
        conditions = [x < 1, (x >= 1) & (x < 2), x >= 2]
        values = [0, 1, 2]
        
        # NumPy backend
        xp.set_backend("numpy")
        result_np = xp.piecewise(x, conditions, values)
        expected = np.array([0, 1, 2])
        np.testing.assert_allclose(result_np, expected)
        
        # PyTorch backend
        try:
            xp.set_backend("pytorch")
            x_torch = xp.array([0.5, 1.5, 2.5], dtype=xp.float64)
            conditions_torch = [x_torch < 1, (x_torch >= 1) & (x_torch < 2), x_torch >= 2]
            result_torch = xp.piecewise(x_torch, conditions_torch, values)
            result_torch_np = xp.to_numpy(result_torch)
            
            np.testing.assert_allclose(result_torch_np, expected, rtol=1e-10)
            
        except ImportError:
            pytest.skip("PyTorch not available")
    
    def test_random_functions(self):
        """Test random number generation functions."""
        # NumPy backend
        xp.set_backend("numpy")
        
        # Test random functions exist and work
        rng = xp.random.default_rng(42)
        assert rng is not None
        
        # PyTorch backend - should raise NotImplementedError for default_rng
        try:
            xp.set_backend("pytorch")
            with pytest.raises(NotImplementedError):
                xp.random.default_rng(42)
        except ImportError:
            pytest.skip("PyTorch not available")
    
    def test_polynomial_functions(self):
        """Test polynomial mathematical functions."""
        # NumPy backend
        xp.set_backend("numpy")
        
        # Test Gauss-Legendre quadrature
        nodes, weights = xp.polynomial.legendre.leggauss(3)
        assert len(nodes) == 3
        assert len(weights) == 3
        
        # PyTorch backend - should raise NotImplementedError
        try:
            xp.set_backend("pytorch")
            with pytest.raises(NotImplementedError):
                xp.polynomial.legendre.leggauss(3)
        except ImportError:
            pytest.skip("PyTorch not available")


if __name__ == "__main__":
    pytest.main([__file__])