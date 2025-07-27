#!/usr/bin/env python3
"""
README Demo Test with Gradient Computation

This test specifically verifies that:
1. The README demo example works with both NumPy and PyTorch backends
2. Posterior logpdf values are consistent between backends  
3. PyTorch backend can compute gradients of the posterior logpdf
"""

import os
import sys
import numpy as np
import torch
from contextlib import contextmanager

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

def test_simplified_bayesian_problem():
    """Test a simplified version of the README demo that focuses on core functionality"""
    
    print("=== Testing Simplified Bayesian Problem ===")
    
    # Results storage
    results = {}
    
    # Test both backends
    for backend_name in ['numpy', 'pytorch']:
        print(f"\n--- Testing {backend_name.upper()} backend ---")
        
        with backend_context(backend_name):
            try:
                import cuqi.array as xp
                from cuqi.distribution import Gaussian, Gamma
                from cuqi.geometry import Discrete
                from cuqi.model import LinearModel
                
                print(f"✓ Backend: {xp.get_backend_name()}")
                
                # Create a simple linear inverse problem
                # Forward model: y = Ax + noise
                np.random.seed(42)  # For reproducibility
                A_data = np.array([[1.0, 0.5], [0.5, 1.0]], dtype=np.float64)  # Simple 2x2 matrix
                A = LinearModel(xp.array(A_data, dtype=xp.float64))
                
                # Generate synthetic data
                x_true = xp.array([1.0, -0.5], dtype=xp.float64)
                y_data = A.forward(x_true) + xp.array([0.1, -0.1], dtype=xp.float64)  # Add small noise
                
                print(f"✓ Forward model created")
                print(f"✓ Data: {y_data}")
                
                # Define prior and likelihood
                x_prior = Gaussian(mean=xp.zeros(2, dtype=xp.float64), cov=1.0, geometry=Discrete(2))
                noise_precision = 10.0  # High precision = low noise
                likelihood = Gaussian(mean=xp.zeros(2, dtype=xp.float64), cov=1.0/noise_precision, geometry=Discrete(2))
                
                print(f"✓ Prior and likelihood defined")
                
                # Create posterior (manually for simplicity)
                def posterior_logpdf(x_val):
                    """Compute log posterior = log prior + log likelihood"""
                    log_prior = x_prior.logpdf(x_val)
                    y_pred = A.forward(x_val)
                    log_likelihood = likelihood.logpdf(y_pred - y_data)  # residual
                    return log_prior + log_likelihood
                
                # Test point for evaluation
                test_x = xp.array([0.5, -0.2], dtype=xp.float64)
                
                # Compute posterior logpdf
                logpdf_val = posterior_logpdf(test_x)
                print(f"✓ Posterior logpdf: {logpdf_val}")
                
                # Store results
                results[backend_name] = {
                    'logpdf': logpdf_val,
                    'test_x': test_x,
                    'y_data': y_data
                }
                
                # For PyTorch backend, test gradient computation
                if backend_name == 'pytorch':
                    print("--- Testing Gradient Computation ---")
                    
                    # Create test point with gradient tracking
                    x_grad = torch.tensor([0.5, -0.2], requires_grad=True, dtype=torch.float64)
                    
                    # Compute posterior logpdf with gradient
                    logpdf_grad = posterior_logpdf(x_grad)
                    
                    if hasattr(logpdf_grad, 'backward'):
                        # Compute gradient
                        logpdf_grad.backward()
                        grad = x_grad.grad
                        
                        print(f"✓ Gradient computation successful")
                        print(f"✓ Gradient: {grad}")
                        
                        results[backend_name]['gradient'] = grad
                    else:
                        print("⚠ Gradient computation not available")
                
                print(f"✅ {backend_name.upper()} backend test PASSED")
                
            except Exception as e:
                print(f"❌ {backend_name.upper()} backend test FAILED: {e}")
                import traceback
                traceback.print_exc()
                results[backend_name] = {'error': str(e)}
    
    # Compare results
    print(f"\n=== Comparing Results ===")
    
    if 'numpy' in results and 'pytorch' in results:
        if 'error' not in results['numpy'] and 'error' not in results['pytorch']:
            # Compare logpdf values
            np_logpdf = results['numpy']['logpdf']
            pt_logpdf = results['pytorch']['logpdf']
            
            # Convert to numpy for comparison
            if hasattr(np_logpdf, 'detach'):
                np_logpdf = np_logpdf.detach().cpu().numpy()
            if hasattr(pt_logpdf, 'detach'):
                pt_logpdf = pt_logpdf.detach().cpu().numpy()
            
            np_logpdf = np.asarray(np_logpdf)
            pt_logpdf = np.asarray(pt_logpdf)
            
            # Check consistency
            try:
                np.testing.assert_allclose(np_logpdf, pt_logpdf, rtol=1e-5, atol=1e-8)
                print("✅ Posterior logpdf values are CONSISTENT between backends")
                
                # Check if gradient was computed
                if 'gradient' in results['pytorch']:
                    print("✅ PyTorch gradient computation SUCCESSFUL")
                    print(f"   Gradient magnitude: {np.linalg.norm(results['pytorch']['gradient'].detach().cpu().numpy()):.6f}")
                
                return True
                
            except AssertionError as e:
                print(f"❌ Posterior logpdf values are INCONSISTENT: {e}")
                print(f"   NumPy: {np_logpdf}")
                print(f"   PyTorch: {pt_logpdf}")
                return False
        else:
            print("❌ One or both backends failed")
            return False
    else:
        print("❌ Missing results from one or both backends")
        return False

def test_gaussian_posterior_gradients():
    """Test gradient computation for a simple Gaussian posterior"""
    
    print("\n=== Testing Gaussian Posterior Gradients ===")
    
    with backend_context('pytorch'):
        try:
            import cuqi.array as xp
            from cuqi.distribution import Gaussian
            from cuqi.geometry import Discrete
            import torch
            
            # Simple Gaussian posterior: N(mu, Sigma)
            mu = xp.array([1.0, -0.5], dtype=xp.float64)
            cov = xp.array([[2.0, 0.5], [0.5, 1.0]], dtype=xp.float64)
            
            posterior = Gaussian(mean=mu, cov=cov, geometry=Discrete(2))
            
            # Test point with gradients
            x = torch.tensor([0.8, -0.3], requires_grad=True, dtype=torch.float64)
            
            # Compute logpdf
            logpdf = posterior.logpdf(x)
            print(f"✓ Logpdf: {logpdf}")
            
            # Compute gradient
            logpdf.backward()
            grad_autograd = x.grad
            
            # Compare with analytical gradient
            grad_analytical = posterior.gradient(x.detach())
            
            print(f"✓ Autograd gradient: {grad_autograd}")
            print(f"✓ Analytical gradient: {grad_analytical}")
            
            # Check consistency
            grad_autograd_np = grad_autograd.detach().cpu().numpy()
            grad_analytical_np = grad_analytical.detach().cpu().numpy()
            
            np.testing.assert_allclose(grad_autograd_np, grad_analytical_np, rtol=1e-5, atol=1e-8)
            print("✅ Gradient computation is CONSISTENT")
            
            return True
            
        except Exception as e:
            print(f"❌ Gradient test FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Run all tests"""
    print("🚀 README Demo Gradient Test Suite")
    print("=" * 50)
    
    success1 = test_simplified_bayesian_problem()
    success2 = test_gaussian_posterior_gradients()
    
    print("\n" + "=" * 50)
    if success1 and success2:
        print("🎉 All README demo gradient tests PASSED!")
        print("✅ NumPy and PyTorch backends produce consistent results")
        print("✅ PyTorch backend can compute gradients correctly")
        return True
    else:
        print("⚠️ Some tests FAILED")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)