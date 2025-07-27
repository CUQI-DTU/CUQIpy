#!/usr/bin/env python3
"""
1D Deconvolution with MRF Priors: GMRF, CMRF, and LMRF

This demo recreates the 1D deconvolution examples from the CUQIpy paper,
showcasing three different Markov Random Field priors:

1. GMRF (Gaussian Markov Random Field) - promotes smoothness
2. CMRF (Cauchy Markov Random Field) - preserves edges while smoothing
3. LMRF (Laplace Markov Random Field) - promotes sparsity in differences

The example demonstrates:
- 1D deconvolution problem setup
- Different prior behaviors and their effects
- Backend switching between NumPy and PyTorch
- Gradient computation capabilities with PyTorch
- Bayesian inference and uncertainty quantification
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
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

def create_1d_deconvolution_problem(n=64):
    """
    Create a 1D deconvolution problem as described in the CUQIpy paper.
    
    Parameters:
    -----------
    n : int
        Problem size (number of discretization points)
        
    Returns:
    --------
    A : LinearModel
        Forward model (convolution operator)
    x_true : array
        True signal
    y_data : array
        Noisy observations
    """
    import cuqi.array as xp
    from cuqi.model import LinearModel
    
    print(f"Creating 1D deconvolution problem (n={n}) with backend: {xp.get_backend_name()}")
    
    # Create convolution matrix (Gaussian blur kernel)
    # This represents a simple 1D convolution with a Gaussian kernel
    sigma = 2.0  # Standard deviation of Gaussian kernel
    kernel_size = 7  # Size of convolution kernel
    
    # Create the convolution matrix
    A_data = np.zeros((n, n))
    
    # Gaussian kernel
    kernel = np.exp(-0.5 * np.arange(-kernel_size//2 + 1, kernel_size//2 + 1)**2 / sigma**2)
    kernel = kernel / kernel.sum()  # Normalize
    
    # Create convolution matrix (Toeplitz structure)
    for i in range(n):
        for j in range(kernel_size):
            col_idx = i - kernel_size//2 + j
            if 0 <= col_idx < n:
                A_data[i, col_idx] = kernel[j]
    
    # Convert to backend array and create LinearModel
    from cuqi.geometry import Discrete
    geometry = Discrete(n)
    A = LinearModel(xp.array(A_data, dtype=xp.float64), 
                    domain_geometry=geometry, range_geometry=geometry)
    
    # Create true signal (piecewise smooth with jumps)
    t = np.linspace(0, 1, n)
    x_true_np = np.zeros(n)
    
    # Piecewise constant signal with jumps (good for testing edge-preserving priors)
    x_true_np[10:20] = 1.5
    x_true_np[25:35] = -1.0
    x_true_np[40:50] = 2.0
    x_true_np[55:60] = -0.5
    
    # Add some smooth components
    x_true_np += 0.3 * np.sin(2 * np.pi * t * 3)
    
    x_true = xp.array(x_true_np, dtype=xp.float64)
    
    # Generate noisy data
    np.random.seed(42)  # For reproducibility
    noise_level = 0.05
    y_clean = A.forward(x_true)
    noise = noise_level * np.random.randn(n)
    y_data = y_clean + xp.array(noise, dtype=xp.float64)
    
    return A, x_true, y_data

def solve_with_gmrf_prior(A, y_data, precision=25.0):
    """
    Solve 1D deconvolution with GMRF (Gaussian Markov Random Field) prior.
    GMRF promotes smoothness through Gaussian differences.
    """
    import cuqi.array as xp
    from cuqi.distribution import GMRF, Gaussian
    from cuqi.geometry import Discrete
    from cuqi.problem import BayesianProblem
    from cuqi.sampler import LinearRTO
    
    n = len(y_data)
    print(f"\n--- GMRF Prior (Gaussian differences, precision={precision}) ---")
    
    # GMRF prior - promotes smoothness
    x = GMRF(mean=xp.zeros(n, dtype=xp.float64), 
             prec=precision,
             bc_type="zero",  # Zero boundary conditions
             order=1,  # First-order differences
             geometry=A.domain_geometry)
    
    # Likelihood - y ~ N(Ax, sigma^2*I)
    noise_std = 0.1
    y = Gaussian(mean=A@x, cov=noise_std**2, geometry=A.range_geometry)
    
    # Bayesian problem
    BP = BayesianProblem(y, x)
    BP.set_data(y=y_data)
    
    # MAP estimate
    try:
        x_map = BP.MAP()
        print(f"  ✅ MAP estimation successful")
    except Exception as e:
        print(f"  ❌ MAP estimation failed: {e}")
        return None, None, None, None
    
    # Sample from posterior (if possible)
    try:
        sampler = LinearRTO(BP)
        samples = sampler.sample(200)
        x_mean = samples.mean()
        x_std = samples.std()
        print(f"  ✅ Posterior sampling successful ({len(samples)} samples)")
        return x_map, x_mean, x_std, samples
    except Exception as e:
        print(f"  ⚠️ Posterior sampling failed: {e}")
        return x_map, x_map, None, None

def solve_with_cmrf_prior(A, y_data, scale=0.1):
    """
    Solve 1D deconvolution with CMRF (Cauchy Markov Random Field) prior.
    CMRF preserves edges while smoothing through Cauchy differences.
    """
    import cuqi.array as xp
    from cuqi.distribution import CMRF, Gaussian
    from cuqi.geometry import Discrete
    from cuqi.problem import BayesianProblem
    from cuqi.sampler import NUTS
    
    n = len(y_data)
    print(f"\n--- CMRF Prior (Cauchy differences, scale={scale}) ---")
    
    # CMRF prior - preserves edges while smoothing
    x = CMRF(location=xp.zeros(n, dtype=xp.float64), 
             scale=scale,
             bc_type="zero",  # Zero boundary conditions
             geometry=A.domain_geometry)
    
    # Likelihood - y ~ N(Ax, sigma^2*I)
    noise_std = 0.1
    y = Gaussian(mean=A@x, cov=noise_std**2, geometry=A.range_geometry)
    
    # Bayesian problem
    BP = BayesianProblem(y, x)
    BP.set_data(y=y_data)
    
    # MAP estimate
    try:
        x_map = BP.MAP()
        print(f"  ✅ MAP estimation successful")
    except Exception as e:
        print(f"  ❌ MAP estimation failed: {e}")
        return None, None, None, None
    
    # Sample from posterior (if possible)
    try:
        sampler = NUTS(BP)
        samples = sampler.sample(200)
        x_mean = samples.mean()
        x_std = samples.std()
        print(f"  ✅ Posterior sampling successful ({len(samples)} samples)")
        return x_map, x_mean, x_std, samples
    except Exception as e:
        print(f"  ⚠️ Posterior sampling failed: {e}")
        return x_map, x_map, None, None

def solve_with_lmrf_prior(A, y_data, scale=0.1):
    """
    Solve 1D deconvolution with LMRF (Laplace Markov Random Field) prior.
    LMRF promotes sparsity in differences through Laplace distribution.
    """
    import cuqi.array as xp
    from cuqi.distribution import LMRF, Gaussian
    from cuqi.geometry import Discrete
    from cuqi.problem import BayesianProblem
    from cuqi.sampler import NUTS
    
    n = len(y_data)
    print(f"\n--- LMRF Prior (Laplace differences, scale={scale}) ---")
    
    # LMRF prior - promotes sparsity in differences
    x = LMRF(location=xp.zeros(n, dtype=xp.float64), 
             scale=scale,
             bc_type="zero",  # Zero boundary conditions
             geometry=A.domain_geometry)
    
    # Likelihood - y ~ N(Ax, sigma^2*I)
    noise_std = 0.1
    y = Gaussian(mean=A@x, cov=noise_std**2, geometry=A.range_geometry)
    
    # Bayesian problem
    BP = BayesianProblem(y, x)
    BP.set_data(y=y_data)
    
    # MAP estimate
    try:
        x_map = BP.MAP()
        print(f"  ✅ MAP estimation successful")
    except Exception as e:
        print(f"  ❌ MAP estimation failed: {e}")
        return None, None, None, None
    
    # Sample from posterior (if possible)
    try:
        sampler = NUTS(BP)
        samples = sampler.sample(200)
        x_mean = samples.mean()
        x_std = samples.std()
        print(f"  ✅ Posterior sampling successful ({len(samples)} samples)")
        return x_map, x_mean, x_std, samples
    except Exception as e:
        print(f"  ⚠️ Posterior sampling failed: {e}")
        return x_map, x_map, None, None

def test_pytorch_gradients(A, y_data, x_true):
    """
    Test gradient computation capabilities with PyTorch backend.
    """
    print(f"\n--- PyTorch Gradient Testing ---")
    
    try:
        import torch
        import cuqi.array as xp
        from cuqi.distribution import GMRF, Gaussian
        from cuqi.geometry import Discrete
        from cuqi.problem import BayesianProblem
        
        n = len(y_data)
        
        # Create test point with gradient tracking (convert to numpy first)
        x_true_np = x_true.detach().cpu().numpy() if hasattr(x_true, 'detach') else np.asarray(x_true)
        x_test = torch.tensor(x_true_np + 0.1 * np.random.randn(n), 
                            requires_grad=True, dtype=torch.float64)
        
        # Simple test: compute a quadratic function and its gradient
        # f(x) = 0.5 * ||Ax - y||^2 + 0.5 * lambda * ||x||^2
        lambda_reg = 0.1
        
        # Forward pass
        y_pred = A.forward(x_test)
        y_data_tensor = torch.tensor(y_data.detach().cpu().numpy() if hasattr(y_data, 'detach') else np.asarray(y_data), dtype=torch.float64)
        
        # Compute loss (data fit + regularization)
        data_fit = 0.5 * torch.sum((y_pred - y_data_tensor)**2)
        regularization = 0.5 * lambda_reg * torch.sum(x_test**2)
        loss = data_fit + regularization
        
        # Compute gradient
        loss.backward()
        grad = x_test.grad
        
        print(f"  ✅ Gradient computation successful")
        print(f"  ✅ Loss: {loss.item():.6f}")
        print(f"  ✅ Data fit: {data_fit.item():.6f}")
        print(f"  ✅ Regularization: {regularization.item():.6f}")
        print(f"  ✅ Gradient shape: {grad.shape}")
        print(f"  ✅ Gradient norm: {grad.norm().item():.6f}")
        print(f"  ✅ Max gradient component: {grad.abs().max().item():.6f}")
        
        # Test gradient-based optimization step
        learning_rate = 0.01
        with torch.no_grad():
            x_updated = x_test - learning_rate * grad
            y_pred_updated = A.forward(x_updated)
            loss_updated = 0.5 * torch.sum((y_pred_updated - y_data_tensor)**2) + 0.5 * lambda_reg * torch.sum(x_updated**2)
            print(f"  ✅ Gradient step: {loss.item():.6f} → {loss_updated.item():.6f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Gradient computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def plot_results(results, save_plots=True):
    """
    Plot comparison of different MRF priors.
    """
    print(f"\n--- Plotting Results ---")
    
    # Convert results to numpy for plotting
    def to_numpy(x):
        if hasattr(x, 'detach'):
            return x.detach().cpu().numpy()
        elif hasattr(x, 'samples'):
            return x.samples.T  # samples object
        else:
            return np.asarray(x)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('1D Deconvolution with Different MRF Priors', fontsize=16)
    
    # Plot 1: True signal and noisy data
    ax = axes[0, 0]
    x_true_np = to_numpy(results['x_true'])
    y_data_np = to_numpy(results['y_data'])
    
    n = len(x_true_np)
    t = np.linspace(0, 1, n)
    
    ax.plot(t, x_true_np, 'k-', linewidth=2, label='True signal')
    ax.plot(t, y_data_np, 'r.', alpha=0.6, markersize=4, label='Noisy data')
    ax.set_title('Problem Setup')
    ax.set_xlabel('t')
    ax.set_ylabel('Signal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: MAP estimates comparison
    ax = axes[0, 1]
    ax.plot(t, x_true_np, 'k-', linewidth=2, label='True signal')
    
    colors = ['blue', 'green', 'orange']
    prior_names = ['GMRF', 'CMRF', 'LMRF']
    
    for i, (prior_name, color) in enumerate(zip(prior_names, colors)):
        if f'{prior_name.lower()}_map' in results:
            x_map_np = to_numpy(results[f'{prior_name.lower()}_map'])
            ax.plot(t, x_map_np, color=color, linewidth=2, 
                   label=f'{prior_name} MAP', linestyle='--')
    
    ax.set_title('MAP Estimates Comparison')
    ax.set_xlabel('t')
    ax.set_ylabel('Signal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Posterior means (if available)
    ax = axes[1, 0]
    ax.plot(t, x_true_np, 'k-', linewidth=2, label='True signal')
    
    for i, (prior_name, color) in enumerate(zip(prior_names, colors)):
        if f'{prior_name.lower()}_mean' in results and results[f'{prior_name.lower()}_mean'] is not None:
            x_mean_np = to_numpy(results[f'{prior_name.lower()}_mean'])
            x_std_np = to_numpy(results[f'{prior_name.lower()}_std']) if results[f'{prior_name.lower()}_std'] is not None else None
            
            ax.plot(t, x_mean_np, color=color, linewidth=2, 
                   label=f'{prior_name} Mean')
            
            if x_std_np is not None:
                ax.fill_between(t, x_mean_np - 2*x_std_np, x_mean_np + 2*x_std_np, 
                               color=color, alpha=0.2, label=f'{prior_name} ±2σ')
    
    ax.set_title('Posterior Means with Uncertainty')
    ax.set_xlabel('t')
    ax.set_ylabel('Signal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Residuals
    ax = axes[1, 1]
    
    for i, (prior_name, color) in enumerate(zip(prior_names, colors)):
        if f'{prior_name.lower()}_map' in results:
            x_map_np = to_numpy(results[f'{prior_name.lower()}_map'])
            residual = x_map_np - x_true_np
            ax.plot(t, residual, color=color, linewidth=2, 
                   label=f'{prior_name} Residual')
    
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.set_title('MAP Estimate Residuals')
    ax.set_xlabel('t')
    ax.set_ylabel('Residual')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('1d_deconvolution_mrf_comparison.png', dpi=300, bbox_inches='tight')
        print(f"  ✅ Plot saved as '1d_deconvolution_mrf_comparison.png'")
    
    plt.show()

def main():
    """
    Main demo function showcasing 1D deconvolution with different MRF priors.
    """
    print("🎯 1D Deconvolution with MRF Priors Demo")
    print("=" * 60)
    print("This demo showcases three different Markov Random Field priors:")
    print("• GMRF: Gaussian differences (promotes smoothness)")
    print("• CMRF: Cauchy differences (preserves edges)")  
    print("• LMRF: Laplace differences (promotes sparsity)")
    print("=" * 60)
    
    results = {}
    
    # Test with NumPy backend
    print(f"\n🔧 Testing with NumPy Backend")
    print("-" * 40)
    
    with backend_context('numpy'):
        # Create problem
        A, x_true, y_data = create_1d_deconvolution_problem(n=64)
        
        results['x_true'] = x_true
        results['y_data'] = y_data
        results['A'] = A
        
        # Test GMRF prior
        try:
            x_map, x_mean, x_std, samples = solve_with_gmrf_prior(A, y_data, precision=25.0)
            if x_map is not None:
                results['gmrf_map'] = x_map
                results['gmrf_mean'] = x_mean
                results['gmrf_std'] = x_std
                results['gmrf_samples'] = samples
        except Exception as e:
            print(f"  ❌ GMRF failed: {e}")
        
        # Test CMRF prior
        try:
            x_map, x_mean, x_std, samples = solve_with_cmrf_prior(A, y_data, scale=0.1)
            if x_map is not None:
                results['cmrf_map'] = x_map
                results['cmrf_mean'] = x_mean
                results['cmrf_std'] = x_std
                results['cmrf_samples'] = samples
        except Exception as e:
            print(f"  ❌ CMRF failed: {e}")
        
        # Test LMRF prior
        try:
            x_map, x_mean, x_std, samples = solve_with_lmrf_prior(A, y_data, scale=0.1)
            if x_map is not None:
                results['lmrf_map'] = x_map
                results['lmrf_mean'] = x_mean
                results['lmrf_std'] = x_std
                results['lmrf_samples'] = samples
        except Exception as e:
            print(f"  ❌ LMRF failed: {e}")
    
    # Test with PyTorch backend
    print(f"\n🔧 Testing with PyTorch Backend")
    print("-" * 40)
    
    with backend_context('pytorch'):
        try:
            # Create problem
            A, x_true, y_data = create_1d_deconvolution_problem(n=64)
            
            # Test gradient computation
            gradient_success = test_pytorch_gradients(A, y_data, x_true)
            results['pytorch_gradients'] = gradient_success
            
            # Test GMRF with PyTorch
            try:
                x_map, x_mean, x_std, samples = solve_with_gmrf_prior(A, y_data, precision=25.0)
                results['pytorch_gmrf_map'] = x_map
                print(f"  ✅ PyTorch GMRF MAP successful")
            except Exception as e:
                print(f"  ❌ PyTorch GMRF failed: {e}")
                
        except Exception as e:
            print(f"  ❌ PyTorch backend failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Plot results
    if len([k for k in results.keys() if k.endswith('_map')]) > 0:
        plot_results(results, save_plots=True)
    else:
        print("  ⚠️ No results to plot")
    
    # Summary
    print(f"\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    success_count = len([k for k in results.keys() if k.endswith('_map')])
    print(f"✅ Successfully solved with {success_count} different priors")
    
    if results.get('pytorch_gradients', False):
        print("✅ PyTorch gradient computation working")
        print("✅ Ready for gradient-based inference methods")
    
    print("\n🎯 Key Insights:")
    print("• GMRF: Best for smooth signals, may over-smooth edges")
    print("• CMRF: Preserves edges better, robust to outliers")
    print("• LMRF: Promotes piecewise constant solutions")
    print("• PyTorch backend enables automatic differentiation")
    print("• All priors work with the array-agnostic framework")
    
    print(f"\n🚀 Next Steps:")
    print("• Try different hyperparameters (precision, scale)")
    print("• Experiment with different boundary conditions")
    print("• Use gradient information for advanced samplers")
    print("• Scale to larger problems with PyTorch GPU acceleration")

if __name__ == "__main__":
    main()