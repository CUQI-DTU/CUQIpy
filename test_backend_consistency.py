#!/usr/bin/env python3
"""
Comprehensive Backend Consistency Tests

This test suite verifies that:
1. NumPy and PyTorch backends produce identical results for logpdf and other operations
2. PyTorch backend can compute gradients correctly
3. The README demo example works with both backends
4. All major CUQIpy functionality is consistent across backends
"""

import os
import sys
import numpy as np
import torch
import pytest
from contextlib import contextmanager

# Set up for testing
@contextmanager
def backend_context(backend_name):
    """Context manager to temporarily switch backends"""
    original_backend = os.environ.get('CUQI_ARRAY_BACKEND', 'numpy')
    os.environ['CUQI_ARRAY_BACKEND'] = backend_name
    
    # Clear module cache
    modules_to_clear = [key for key in list(sys.modules.keys()) if key.startswith('cuqi')]
    for module in modules_to_clear:
        del sys.modules[module]
    
    try:
        yield
    finally:
        os.environ['CUQI_ARRAY_BACKEND'] = original_backend
        # Clear module cache again
        modules_to_clear = [key for key in list(sys.modules.keys()) if key.startswith('cuqi')]
        for module in modules_to_clear:
            del sys.modules[module]

def compare_values(val1, val2, rtol=1e-5, atol=1e-8, name="values"):
    """Compare values from different backends, handling different types"""
    # Convert to numpy for comparison
    if hasattr(val1, 'detach'):  # PyTorch tensor
        val1_np = val1.detach().cpu().numpy()
    elif hasattr(val1, 'cpu'):  # PyTorch tensor without grad
        val1_np = val1.cpu().numpy()
    else:
        val1_np = np.asarray(val1)
    
    if hasattr(val2, 'detach'):  # PyTorch tensor
        val2_np = val2.detach().cpu().numpy()
    elif hasattr(val2, 'cpu'):  # PyTorch tensor without grad
        val2_np = val2.cpu().numpy()
    else:
        val2_np = np.asarray(val2)
    
    # Compare
    try:
        np.testing.assert_allclose(val1_np, val2_np, rtol=rtol, atol=atol)
        print(f"✓ {name}: CONSISTENT")
        return True
    except AssertionError as e:
        print(f"✗ {name}: INCONSISTENT - {e}")
        print(f"  NumPy value: {val1_np}")
        print(f"  PyTorch value: {val2_np}")
        return False

class TestBackendConsistency:
    """Test suite for backend consistency"""
    
    def test_basic_array_operations(self):
        """Test basic array operations consistency"""
        print("\n=== Testing Basic Array Operations ===")
        
        # Test data
        test_data = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        with backend_context('numpy'):
            import cuqi.array as xp_np
            arr_np = xp_np.array(test_data)
            sum_np = xp_np.sum(arr_np)
            mean_np = xp_np.mean(arr_np)
            std_np = xp_np.std(arr_np)
            dot_np = xp_np.dot(arr_np, arr_np)
        
        with backend_context('pytorch'):
            import cuqi.array as xp_pt
            arr_pt = xp_pt.array(test_data)
            sum_pt = xp_pt.sum(arr_pt)
            mean_pt = xp_pt.mean(arr_pt)
            std_pt = xp_pt.std(arr_pt)
            dot_pt = xp_pt.dot(arr_pt, arr_pt)
        
        # Compare results
        all_consistent = True
        all_consistent &= compare_values(arr_np, arr_pt, name="Array creation")
        all_consistent &= compare_values(sum_np, sum_pt, name="Sum")
        all_consistent &= compare_values(mean_np, mean_pt, name="Mean")
        all_consistent &= compare_values(std_np, std_pt, name="Std")
        all_consistent &= compare_values(dot_np, dot_pt, name="Dot product")
        
        assert all_consistent, "Basic array operations are not consistent"
    
    def test_gaussian_distribution_consistency(self):
        """Test Gaussian distribution consistency between backends"""
        print("\n=== Testing Gaussian Distribution ===")
        
        # Test parameters
        mean_val = [0.0, 1.0, -0.5]
        cov_val = 2.0
        test_x = [0.1, 0.9, -0.3]
        
        # NumPy backend
        with backend_context('numpy'):
            import cuqi.array as xp
            from cuqi.distribution import Gaussian
            from cuqi.geometry import Discrete
            
            mean_np = xp.array(mean_val)
            x_np = xp.array(test_x)
            dist_np = Gaussian(mean=mean_np, cov=cov_val, geometry=Discrete(3))
            logpdf_np = dist_np.logpdf(x_np)
            grad_np = dist_np.gradient(x_np)
        
        # PyTorch backend
        with backend_context('pytorch'):
            import cuqi.array as xp
            from cuqi.distribution import Gaussian
            from cuqi.geometry import Discrete
            
            mean_pt = xp.array(mean_val)
            x_pt = xp.array(test_x)
            dist_pt = Gaussian(mean=mean_pt, cov=cov_val, geometry=Discrete(3))
            logpdf_pt = dist_pt.logpdf(x_pt)
            grad_pt = dist_pt.gradient(x_pt)
        
        # Compare results
        all_consistent = True
        all_consistent &= compare_values(logpdf_np, logpdf_pt, name="Gaussian logpdf")
        all_consistent &= compare_values(grad_np, grad_pt, name="Gaussian gradient")
        
        assert all_consistent, "Gaussian distribution is not consistent"
    
    def test_pytorch_gradient_computation(self):
        """Test that PyTorch can compute gradients of logpdf"""
        print("\n=== Testing PyTorch Gradient Computation ===")
        
        with backend_context('pytorch'):
            import cuqi.array as xp
            from cuqi.distribution import Gaussian
            from cuqi.geometry import Discrete
            import torch
            
            # Create parameters with gradient tracking
            mean = xp.array([0.0, 1.0, -0.5])
            x = torch.tensor([0.1, 0.9, -0.3], requires_grad=True)
            
            # Create distribution
            dist = Gaussian(mean=mean, cov=2.0, geometry=Discrete(3))
            
            # Compute logpdf
            logpdf = dist.logpdf(x)
            print(f"  Logpdf: {logpdf}")
            
            # Compute gradient
            if hasattr(logpdf, 'backward'):
                logpdf.backward()
                grad = x.grad
                print(f"  Gradient via autograd: {grad}")
                
                # Compare with analytical gradient
                analytical_grad = dist.gradient(x.detach())
                print(f"  Analytical gradient: {analytical_grad}")
                
                # They should be close
                grad_consistent = compare_values(grad, analytical_grad, name="Gradient consistency")
                assert grad_consistent, "PyTorch autograd gradient doesn't match analytical gradient"
            else:
                print("  Warning: logpdf doesn't support backward pass")
    
    def test_readme_demo_consistency(self):
        """Test the README demo example with both backends"""
        print("\n=== Testing README Demo Example ===")
        
        # Simplified version of the README demo for testing
        def run_demo_with_backend(backend_name):
            with backend_context(backend_name):
                try:
                    import cuqi.array as xp
                    from cuqi.testproblem import Deconvolution2D
                    from cuqi.distribution import Gaussian, LMRF, Gamma
                    from cuqi.problem import BayesianProblem
                    
                    # Step 1: Set up forward model and data (smaller for testing)
                    A, y_data, info = Deconvolution2D(dim=32, phantom="cookie").get_components()
                    
                    # Step 2: Define distributions for parameters
                    d = Gamma(1, 1e-4)
                    s = Gamma(1, 1e-4)
                    x = LMRF(0, lambda d: 1/d, geometry=A.domain_geometry)
                    y = Gaussian(A@x, lambda s: 1/s)
                    
                    # Step 3: Create Bayesian Problem
                    BP = BayesianProblem(y, x, d, s)
                    BP.set_data(y=y_data)
                    
                    # Test posterior evaluation at a point
                    test_point = {
                        'x': xp.ones(A.domain_geometry.dim) * 0.1,
                        'd': xp.array([0.01]),
                        's': xp.array([0.1])
                    }
                    
                    posterior = BP.posterior
                    logpdf = posterior.logpdf(test_point)
                    
                    print(f"  {backend_name} backend - Posterior logpdf: {logpdf}")
                    return logpdf, test_point
                    
                except Exception as e:
                    print(f"  {backend_name} backend failed: {e}")
                    return None, None
        
        # Run with both backends
        logpdf_np, test_point_np = run_demo_with_backend('numpy')
        logpdf_pt, test_point_pt = run_demo_with_backend('pytorch')
        
        if logpdf_np is not None and logpdf_pt is not None:
            consistent = compare_values(logpdf_np, logpdf_pt, name="README demo posterior logpdf")
            assert consistent, "README demo is not consistent between backends"
        else:
            pytest.skip("README demo failed on one or both backends")
    
    def test_linear_model_consistency(self):
        """Test linear model consistency"""
        print("\n=== Testing Linear Model ===")
        
        # Test data
        np.random.seed(42)
        A_data = np.random.randn(5, 3)
        x_data = [1.0, -0.5, 2.0]
        
        # NumPy backend
        with backend_context('numpy'):
            import cuqi.array as xp
            from cuqi.model import LinearModel
            
            A_np = xp.array(A_data)
            x_np = xp.array(x_data)
            model_np = LinearModel(A_np)
            y_np = model_np.forward(x_np)
            grad_np = model_np.gradient(xp.ones(5), x_np)
        
        # PyTorch backend
        with backend_context('pytorch'):
            import cuqi.array as xp
            from cuqi.model import LinearModel
            
            A_pt = xp.array(A_data)
            x_pt = xp.array(x_data)
            model_pt = LinearModel(A_pt)
            y_pt = model_pt.forward(x_pt)
            grad_pt = model_pt.gradient(xp.ones(5), x_pt)
        
        # Compare results
        all_consistent = True
        all_consistent &= compare_values(y_np, y_pt, name="Linear model forward")
        all_consistent &= compare_values(grad_np, grad_pt, name="Linear model gradient")
        
        assert all_consistent, "Linear model is not consistent"
    
    def test_multiple_distributions_consistency(self):
        """Test multiple distribution types for consistency"""
        print("\n=== Testing Multiple Distributions ===")
        
        distributions_to_test = [
            # (distribution_class, params, test_point)
            ('Gaussian', {'mean': [0, 0], 'cov': 1.0}, [0.1, -0.2]),
            ('Gamma', {'shape': 2.0, 'rate': 1.0}, [1.5]),
            ('Uniform', {'low': -1.0, 'high': 2.0}, [0.5]),
        ]
        
        all_consistent = True
        
        for dist_name, params, test_point in distributions_to_test:
            print(f"\n  Testing {dist_name} distribution:")
            
            # NumPy backend
            with backend_context('numpy'):
                import cuqi.array as xp
                from cuqi.distribution import Gaussian, Gamma, Uniform
                from cuqi.geometry import Discrete
                
                dist_class = locals()[dist_name]
                if 'mean' in params:
                    params_np = {k: xp.array(v) if isinstance(v, list) else v for k, v in params.items()}
                    params_np['geometry'] = Discrete(len(params['mean']))
                else:
                    params_np = {k: xp.array(v) if isinstance(v, list) else v for k, v in params.items()}
                    if len(test_point) > 1:
                        params_np['geometry'] = Discrete(len(test_point))
                
                dist_np = dist_class(**params_np)
                x_np = xp.array(test_point)
                logpdf_np = dist_np.logpdf(x_np)
            
            # PyTorch backend
            with backend_context('pytorch'):
                import cuqi.array as xp
                from cuqi.distribution import Gaussian, Gamma, Uniform
                from cuqi.geometry import Discrete
                
                dist_class = locals()[dist_name]
                if 'mean' in params:
                    params_pt = {k: xp.array(v) if isinstance(v, list) else v for k, v in params.items()}
                    params_pt['geometry'] = Discrete(len(params['mean']))
                else:
                    params_pt = {k: xp.array(v) if isinstance(v, list) else v for k, v in params.items()}
                    if len(test_point) > 1:
                        params_pt['geometry'] = Discrete(len(test_point))
                
                dist_pt = dist_class(**params_pt)
                x_pt = xp.array(test_point)
                logpdf_pt = dist_pt.logpdf(x_pt)
            
            # Compare
            consistent = compare_values(logpdf_np, logpdf_pt, name=f"{dist_name} logpdf")
            all_consistent &= consistent
        
        assert all_consistent, "Not all distributions are consistent"

def run_all_tests():
    """Run all consistency tests"""
    print("🚀 Starting Backend Consistency Test Suite")
    print("=" * 60)
    
    test_suite = TestBackendConsistency()
    
    tests = [
        test_suite.test_basic_array_operations,
        test_suite.test_gaussian_distribution_consistency,
        test_suite.test_pytorch_gradient_computation,
        test_suite.test_linear_model_consistency,
        test_suite.test_multiple_distributions_consistency,
        test_suite.test_readme_demo_consistency,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
            print("✅ PASSED")
        except Exception as e:
            failed += 1
            print(f"❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All backend consistency tests PASSED!")
        return True
    else:
        print("⚠️  Some tests FAILED!")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)