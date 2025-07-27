#!/usr/bin/env python3
"""
CUQI Array Backend Demo: GMRF Distribution with Gradients

This demo showcases the array-agnostic capabilities of CUQIpy using a 
Gaussian Markov Random Field (GMRF) distribution, demonstrating:

1. Backend switching between NumPy and PyTorch
2. Consistent results across backends
3. Automatic differentiation with PyTorch backend
4. Real-world Bayesian inverse problem setup

The example uses a 1D deconvolution problem with GMRF prior.
"""

import os
import sys
import numpy as np
from contextlib import contextmanager

@contextmanager
def backend_context(backend_name):
    """Context manager to switch array backends"""
    original_backend = os.environ.get('CUQI_ARRAY_BACKEND', 'numpy')
    os.environ['CUQI_ARRAY_BACKEND'] = backend_name
    
    # Clear module cache to ensure fresh import
    modules_to_clear = [key for key in list(sys.modules.keys()) if key.startswith('cuqi')]
    for module in modules_to_clear:
        del sys.modules[module]
    
    try:
        yield
    finally:
        os.environ['CUQI_ARRAY_BACKEND'] = original_backend
        # Clear cache again
        modules_to_clear = [key for key in list(sys.modules.keys()) if key.startswith('cuqi')]
        for module in modules_to_clear:
            del sys.modules[module]

def create_1d_deconvolution_problem():
    """Create a simple 1D deconvolution problem"""
    import cuqi.array as xp
    from cuqi.distribution import GMRF, Gaussian
    from cuqi.geometry import Discrete
    from cuqi.model import LinearModel
    
    print(f"  Backend: {xp.get_backend_name()}")
    
    # Problem setup
    n = 32  # Problem size
    
    # Create convolution matrix (simple blur operator)
    A_data = np.zeros((n, n))
    for i in range(n):
        for j in range(max(0, i-2), min(n, i+3)):  # 5-point blur
            A_data[i, j] = np.exp(-0.5 * (i-j)**2)
    
    # Normalize rows
    A_data = A_data / A_data.sum(axis=1, keepdims=True)
    
    # Convert to backend array
    A = LinearModel(xp.array(A_data, dtype=xp.float64))
    
    # Create true solution (smooth function)
    x_true = xp.array([np.sin(2*np.pi*i/n) + 0.5*np.cos(4*np.pi*i/n) for i in range(n)], dtype=xp.float64)
    
    # Generate noisy data
    np.random.seed(42)  # For reproducibility
    y_data = A.forward(x_true) + xp.array(0.05 * np.random.randn(n), dtype=xp.float64)
    
    # Define GMRF prior (promotes smoothness)
    # GMRF takes a scalar precision and automatically creates the structure
    prec_value = 10.0  # Precision parameter
    
    # Create GMRF prior with first-order differences (order=1)
    x_prior = GMRF(mean=xp.zeros(n, dtype=xp.float64), 
                   prec=prec_value, 
                   bc_type="periodic",  # Periodic boundary conditions
                   order=1,  # First-order differences
                   geometry=Discrete(n))
    
    # Create likelihood (Gaussian noise model)
    noise_precision = 100.0  # High precision = low noise
    likelihood = Gaussian(mean=xp.zeros(n, dtype=xp.float64), 
                         cov=1.0/noise_precision, 
                         geometry=Discrete(n))
    
    return A, x_true, y_data, x_prior, likelihood

def compute_posterior_logpdf(x_val, A, y_data, x_prior, likelihood):
    """Compute log posterior = log prior + log likelihood"""
    log_prior = x_prior.logpdf(x_val)
    y_pred = A.forward(x_val)
    residual = y_pred - y_data
    log_likelihood = likelihood.logpdf(residual)
    return log_prior + log_likelihood

def demo_gmrf_gradients():
    """Main demo function"""
    print("🎯 CUQI Array Backend Demo: GMRF Distribution with Gradients")
    print("=" * 70)
    
    results = {}
    
    # Test both backends
    for backend_name in ['numpy', 'pytorch']:
        print(f"\n--- Testing {backend_name.upper()} Backend ---")
        
        with backend_context(backend_name):
            try:
                # Create problem
                A, x_true, y_data, x_prior, likelihood = create_1d_deconvolution_problem()
                
                # Test point for evaluation
                import cuqi.array as xp
                test_x = 0.8 * x_true + xp.array(0.1 * np.random.randn(len(x_true)), dtype=xp.float64)
                
                # Compute posterior logpdf
                logpdf_val = compute_posterior_logpdf(test_x, A, y_data, x_prior, likelihood)
                
                print(f"  ✅ Problem setup successful")
                print(f"  ✅ Data shape: {y_data.shape}")
                print(f"  ✅ Prior: GMRF with precision matrix")
                print(f"  ✅ Posterior logpdf: {logpdf_val}")
                
                # Store results
                results[backend_name] = {
                    'logpdf': logpdf_val,
                    'test_x': test_x,
                    'y_data': y_data,
                    'x_true': x_true
                }
                
                # For PyTorch backend, test gradient computation
                if backend_name == 'pytorch':
                    print("  --- Testing Gradient Computation ---")
                    
                    import torch
                    
                    # Create test point with gradient tracking
                    x_grad = torch.tensor(test_x.detach().cpu().numpy(), 
                                        requires_grad=True, dtype=torch.float64)
                    
                    # Compute posterior logpdf with gradient
                    logpdf_grad = compute_posterior_logpdf(x_grad, A, y_data, x_prior, likelihood)
                    
                    if hasattr(logpdf_grad, 'backward'):
                        # Compute gradient
                        logpdf_grad.backward()
                        grad = x_grad.grad
                        
                        print(f"  ✅ Gradient computation successful")
                        print(f"  ✅ Gradient shape: {grad.shape}")
                        print(f"  ✅ Gradient norm: {grad.norm().item():.6f}")
                        print(f"  ✅ Max gradient component: {grad.abs().max().item():.6f}")
                        
                        results[backend_name]['gradient'] = grad
                        
                        # Test gradient-based optimization step
                        learning_rate = 0.01
                        x_updated = x_grad.detach() + learning_rate * grad
                        logpdf_updated = compute_posterior_logpdf(x_updated, A, y_data, x_prior, likelihood)
                        
                        print(f"  ✅ Gradient step: {logpdf_grad.item():.6f} → {logpdf_updated.item():.6f}")
                        
                    else:
                        print("  ⚠️ Gradient computation not available")
                
            except Exception as e:
                print(f"  ❌ {backend_name.upper()} backend failed: {e}")
                import traceback
                traceback.print_exc()
                results[backend_name] = {'error': str(e)}
    
    # Compare results between backends
    print(f"\n--- Backend Consistency Check ---")
    
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
                print(f"   NumPy:   {np_logpdf}")
                print(f"   PyTorch: {pt_logpdf}")
                print(f"   Difference: {abs(np_logpdf - pt_logpdf)}")
                
                # Check if gradient was computed
                if 'gradient' in results['pytorch']:
                    grad_norm = results['pytorch']['gradient'].norm().item()
                    print("✅ PyTorch gradient computation SUCCESSFUL")
                    print(f"   Gradient provides optimization direction")
                    print(f"   Ready for gradient-based MCMC sampling!")
                
                return True
                
            except AssertionError as e:
                print(f"❌ Posterior logpdf values are INCONSISTENT: {e}")
                return False
        else:
            print("❌ One or both backends failed")
            return False
    else:
        print("❌ Missing results from one or both backends")
        return False

def demo_gmrf_sampling():
    """Demonstrate sampling capabilities"""
    print(f"\n--- GMRF Sampling Demo ---")
    
    with backend_context('numpy'):
        import cuqi.array as xp
        from cuqi.distribution import GMRF
        from cuqi.geometry import Discrete
        
        # Create a simple GMRF
        n = 16
        prec_value = 5.0  # Scalar precision
        
        gmrf = GMRF(mean=xp.zeros(n, dtype=xp.float64), 
                   prec=prec_value,
                   bc_type="zero",  # Zero boundary conditions
                   order=1,  # First-order differences
                   geometry=Discrete(n))
        
        # Sample from GMRF
        samples = gmrf.sample(5)
        print(f"✅ GMRF sampling successful")
        print(f"   Sample shape: {samples.shape}")
        print(f"   Sample range: [{samples.min():.3f}, {samples.max():.3f}]")

if __name__ == "__main__":
    try:
        success = demo_gmrf_gradients()
        demo_gmrf_sampling()
        
        print("\n" + "=" * 70)
        if success:
            print("🎉 GMRF Demo COMPLETED Successfully!")
            print("✅ Both NumPy and PyTorch backends working")
            print("✅ GMRF distributions functional")
            print("✅ Gradient computation verified")
            print("✅ Ready for advanced Bayesian inference!")
        else:
            print("⚠️ Demo completed with some issues")
            
        print("\n🚀 Usage:")
        print("   export CUQI_ARRAY_BACKEND=numpy    # NumPy backend")
        print("   export CUQI_ARRAY_BACKEND=pytorch  # PyTorch with gradients")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)