#!/usr/bin/env python3
"""
CUQIpy Array Backends Comprehensive Demo
========================================

This demo provides a comprehensive overview of CUQIpy's array-agnostic framework,
showcasing how to leverage different computational backends for Bayesian inference.

Features Demonstrated:
1. **Backend Selection**: NumPy, PyTorch, CuPy, JAX support
2. **Performance Comparison**: Speed and memory considerations
3. **Automatic Differentiation**: PyTorch gradient computation
4. **GPU Acceleration**: CuPy and PyTorch GPU capabilities
5. **Numerical Consistency**: Verification across backends
6. **Best Practices**: When to use which backend

This demo serves as both a tutorial and a benchmark for understanding
the capabilities and trade-offs of different array backends in CUQIpy.
"""

import sys
sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt
import time

import cuqi
import cuqi.array as xp
from cuqi.distribution import Gaussian, GMRF
from cuqi.model import LinearModel
from cuqi.problem import BayesianProblem

print("🚀 CUQIpy Array Backends - Comprehensive Demo")
print("=" * 50)
print("Exploring the power of array-agnostic Bayesian inference\n")

# %%
# ## 1. Backend Overview and Selection

def show_backend_info():
    """Display information about available backends."""
    print("📋 Available Array Backends:")
    print("-" * 30)
    
    backends = ["numpy", "pytorch", "cupy", "jax"]
    available = []
    
    for backend in backends:
        try:
            xp.set_backend(backend)
            available.append(backend)
            print(f"  ✅ {backend.title()}: Available")
        except ImportError:
            print(f"  ❌ {backend.title()}: Not installed")
        except Exception as e:
            print(f"  ⚠️  {backend.title()}: Error - {e}")
    
    # Reset to numpy
    xp.set_backend("numpy")
    return available

available_backends = show_backend_info()

# %%
# ## 2. Problem Setup

print(f"\n🔧 Setting up test problem...")

# Create a simple deconvolution problem
n = 64  # Problem size
np.random.seed(42)

# Forward model matrix
A_data = np.random.randn(n, n) * 0.1 + np.eye(n)

# True signal (piecewise constant)
x_true = np.zeros(n)
x_true[15:25] = 1.5
x_true[35:45] = -1.0

# Observed data
y_data = A_data @ x_true + 0.05 * np.random.randn(n)

print(f"  ✓ Problem size: {n}x{n}")
print(f"  ✓ Signal sparsity: {np.sum(x_true != 0)} non-zero elements")
print(f"  ✓ Noise level: 5%")

# %%
# ## 3. Backend Performance Comparison

def benchmark_backend(backend_name, problem_size=n):
    """Benchmark a specific backend."""
    if backend_name not in available_backends:
        return None
    
    print(f"\n🔬 Benchmarking {backend_name.upper()} backend:")
    
    try:
        # Switch backend
        start_time = time.time()
        xp.set_backend(backend_name)
        
        # Problem setup
        A = LinearModel(xp.array(A_data, dtype=xp.float64))
        x = Gaussian(mean=xp.zeros(problem_size, dtype=xp.float64), cov=1.0)
        y = Gaussian(mean=A@x, cov=0.01)
        BP = BayesianProblem(y, x)
        BP.set_data(y=xp.array(y_data, dtype=xp.float64))
        
        setup_time = time.time() - start_time
        
        # Benchmark logpdf computation
        x_test = xp.array(np.random.randn(problem_size), dtype=xp.float64)
        
        start_time = time.time()
        for _ in range(10):  # Multiple evaluations for timing
            logpdf = BP.posterior.logpdf(x_test)
        logpdf_time = (time.time() - start_time) / 10
        
        # Test gradient computation if PyTorch
        grad_time = None
        if backend_name == "pytorch":
            x_grad = xp.array(np.random.randn(problem_size), requires_grad=True, dtype=xp.float64)
            start_time = time.time()
            logpdf_grad = BP.posterior.logpdf(x_grad)
            logpdf_grad.backward()
            grad_time = time.time() - start_time
        
        results = {
            'backend': backend_name,
            'setup_time': setup_time,
            'logpdf_time': logpdf_time,
            'grad_time': grad_time,
            'logpdf_value': float(logpdf.item() if hasattr(logpdf, 'item') else logpdf),
            'success': True
        }
        
        print(f"  ✓ Setup time: {setup_time*1000:.2f} ms")
        print(f"  ✓ LogPDF time: {logpdf_time*1000:.2f} ms")
        print(f"  ✓ LogPDF value: {results['logpdf_value']:.4f}")
        if grad_time is not None:
            print(f"  ✓ Gradient time: {grad_time*1000:.2f} ms")
        
        return results
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return {'backend': backend_name, 'success': False, 'error': str(e)}

# Run benchmarks
benchmark_results = []
for backend in available_backends:
    result = benchmark_backend(backend)
    if result:
        benchmark_results.append(result)

# %%
# ## 4. Numerical Consistency Check

print(f"\n🔍 Numerical Consistency Check:")
print("-" * 35)

if len(benchmark_results) >= 2:
    successful_results = [r for r in benchmark_results if r['success']]
    
    if len(successful_results) >= 2:
        reference = successful_results[0]
        print(f"Reference: {reference['backend'].upper()} = {reference['logpdf_value']:.8f}")
        
        for result in successful_results[1:]:
            diff = abs(result['logpdf_value'] - reference['logpdf_value'])
            consistent = diff < 1e-6
            status = "✅" if consistent else "❌"
            print(f"{status} {result['backend'].upper()}: diff = {diff:.2e} ({'OK' if consistent else 'FAIL'})")
    else:
        print("⚠️  Not enough successful backends for comparison")
else:
    print("⚠️  Not enough backends available for comparison")

# %%
# ## 5. PyTorch Automatic Differentiation Showcase

if "pytorch" in available_backends:
    print(f"\n🧠 PyTorch Automatic Differentiation:")
    print("-" * 40)
    
    xp.set_backend("pytorch")
    
    # Set up problem
    A_torch = LinearModel(xp.array(A_data, dtype=xp.float64))
    x_prior = Gaussian(mean=xp.zeros(n, dtype=xp.float64), cov=1.0)
    y_like = Gaussian(mean=A_torch@x_prior, cov=0.01)
    BP_torch = BayesianProblem(y_like, x_prior)
    BP_torch.set_data(y=xp.array(y_data, dtype=xp.float64))
    
    # Demonstrate gradient-based optimization
    print("  🎯 Gradient-based MAP estimation:")
    
    # Initial guess
    x_init = xp.array(np.random.randn(n) * 0.1, requires_grad=True, dtype=xp.float64)
    
    # Simple gradient ascent (few steps for demo)
    learning_rate = 0.001
    for i in range(5):
        # Compute log posterior
        logpdf = BP_torch.posterior.logpdf(x_init)
        
        # Backward pass
        if x_init.grad is not None:
            x_init.grad.zero_()
        logpdf.backward()
        
        # Gradient ascent step (maximize log posterior)
        with xp._backend_module.no_grad():
            x_init += learning_rate * x_init.grad
        
        print(f"    Step {i+1}: logpdf = {logpdf.item():.4f}, grad_norm = {xp.linalg.norm(x_init.grad).item():.4f}")
    
    print("  ✅ Gradient-based optimization completed")

# %%
# ## 6. Performance Summary and Recommendations

print(f"\n📊 Performance Summary:")
print("=" * 25)

if benchmark_results:
    # Create performance table
    print(f"{'Backend':<10} {'Setup (ms)':<12} {'LogPDF (ms)':<12} {'Gradient (ms)':<12} {'Status'}")
    print("-" * 60)
    
    for result in benchmark_results:
        if result['success']:
            setup = f"{result['setup_time']*1000:.2f}"
            logpdf = f"{result['logpdf_time']*1000:.2f}"
            grad = f"{result['grad_time']*1000:.2f}" if result['grad_time'] else "N/A"
            status = "✅"
        else:
            setup = logpdf = grad = "Failed"
            status = "❌"
        
        print(f"{result['backend'].title():<10} {setup:<12} {logpdf:<12} {grad:<12} {status}")

print(f"\n💡 Recommendations:")
print("-" * 20)
print("🔹 **NumPy**: Default choice, maximum compatibility, CPU-optimized")
print("🔹 **PyTorch**: Best for gradient-based methods (HMC, NUTS, MALA)")
print("🔹 **CuPy**: GPU acceleration for large linear algebra operations")
print("🔹 **JAX**: JIT compilation, functional programming, research applications")

print(f"\n🎯 Backend Selection Guide:")
print("-" * 25)
print("• Small problems (< 1000 params): NumPy")
print("• Gradient-based sampling: PyTorch")
print("• Large matrix operations: CuPy")
print("• Performance-critical research: JAX")

# %%
# ## 7. Visualization

if len(benchmark_results) >= 2:
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Setup time comparison
    plt.subplot(2, 2, 1)
    successful = [r for r in benchmark_results if r['success']]
    backends = [r['backend'] for r in successful]
    setup_times = [r['setup_time']*1000 for r in successful]
    
    plt.bar(backends, setup_times, alpha=0.7, color='skyblue')
    plt.title('Setup Time Comparison')
    plt.ylabel('Time (ms)')
    plt.xticks(rotation=45)
    
    # Plot 2: LogPDF computation time
    plt.subplot(2, 2, 2)
    logpdf_times = [r['logpdf_time']*1000 for r in successful]
    
    plt.bar(backends, logpdf_times, alpha=0.7, color='lightgreen')
    plt.title('LogPDF Computation Time')
    plt.ylabel('Time (ms)')
    plt.xticks(rotation=45)
    
    # Plot 3: Problem visualization
    plt.subplot(2, 2, 3)
    plt.plot(x_true, 'k-', linewidth=2, label='True signal')
    plt.plot(y_data, 'r.', alpha=0.6, label='Observed data')
    plt.title('Test Problem')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Numerical consistency
    plt.subplot(2, 2, 4)
    if len(successful) >= 2:
        ref_value = successful[0]['logpdf_value']
        values = [r['logpdf_value'] for r in successful]
        diffs = [abs(v - ref_value) for v in values]
        
        plt.semilogy(backends, diffs, 'o-', linewidth=2, markersize=8)
        plt.axhline(y=1e-6, color='r', linestyle='--', alpha=0.7, label='Tolerance (1e-6)')
        plt.title('Numerical Consistency')
        plt.ylabel('Absolute Difference')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('array_backends_benchmark.png', dpi=150, bbox_inches='tight')
    print(f"\n📈 Benchmark plot saved as 'array_backends_benchmark.png'")

# Reset to numpy backend
xp.set_backend("numpy")

print(f"\n🎉 Array backends demo completed!")
print("🔗 For more information, see: docs/user/array_backends.rst")