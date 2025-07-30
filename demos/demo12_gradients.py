# %%
import sys
sys.path.append("../")
import numpy as np
import matplotlib.pyplot as plt
import cuqi
import cuqi.array as xp

print(" Gradient Computation with Array Backends")
print("=" * 45)
print("This demo showcases gradient computation capabilities")
print("across different array backends, especially PyTorch.\n")

# %%
# Load cuqi deblur model and data
TP = cuqi.testproblem._Deblur()
n = TP.model.domain_dim
m = TP.model.range_dim
x_true = TP.exactSolution

# %%
# Compare gradient computation across backends
def test_gradients_with_backend(backend_name):
    """Test gradient computation with specified backend."""
    print(f"\n Testing with {backend_name.upper()} backend:")
    
    try:
        xp.set_backend(backend_name)
        print(f"  ✓ Backend set: {xp.get_backend_name()}")
        
        # Define prior using array backend
        var = 0.2
        prior = cuqi.distribution.Gaussian(xp.zeros(n), var)
        
        # Define likelihood
        likelihood = cuqi.distribution.Gaussian(TP.model@prior, 0.1)
        
        # Create posterior
        posterior = cuqi.distribution.Posterior(likelihood, prior)
        
        # Test point for gradient computation
        x_test = xp.array(np.random.randn(n), dtype=xp.float64)
        
        if backend_name == "pytorch":
            # Enable gradient tracking for PyTorch
            x_test = xp.array(np.random.randn(n), requires_grad=True, dtype=xp.float64)
            
            # Compute log posterior
            logpdf = posterior.logpdf(x_test)
            print(f"  ✓ Log posterior: {logpdf.item():.4f}")
            
            # Compute gradients using automatic differentiation
            logpdf.backward()
            grad_norm = xp.linalg.norm(x_test.grad).item()
            print(f"  ✓ Automatic gradient norm: {grad_norm:.4f}")
            
        else:
            # Use finite differences for other backends
            logpdf = posterior.logpdf(x_test)
            print(f"  ✓ Log posterior: {float(logpdf):.4f}")
            
            # Finite difference gradient (approximation)
            grad_fd = posterior.gradient(x_test)
            grad_norm = xp.linalg.norm(grad_fd)
            print(f"  ✓ Finite difference gradient norm: {float(grad_norm):.4f}")
            
    except ImportError:
        print(f"  ⚠️  {backend_name.title()} not available")
    except Exception as e:
        print(f"   Error with {backend_name}: {e}")

# Test gradients with different backends
test_gradients_with_backend("numpy")
test_gradients_with_backend("pytorch")

# Reset to numpy for rest of demo
xp.set_backend("numpy")

# %%
# Original demo continues with numpy backend
pr = 'gaussian'
if (pr == 'gaussian'):
    var = 0.2
    prior = cuqi.distribution.Gaussian(xp.zeros(n), var)
elif (pr == 'cauchy'):
    h = 1/n
    delta = 0.3
    prior = cuqi.distribution.CMRF(xp.zeros(n), delta*h, 'neumann')

# %%
# Compare with MAP computed using BayesianProblem (within the testproblem)
x0 = np.random.randn(n)
TP.prior = prior
if (pr == 'gaussian'):
    x_MAP = TP.MAP()
else:
    # Solve posterior problem using BFGS
    def f(x): return -TP.posterior.logd(x)
    def gradf(x): return -TP.posterior.gradient(x)
    solver = cuqi.solver.ScipyLBFGSB(f, x0, gradf)
    x_MAP, solution_info = solver.solve()
print('relative error MAP:', np.linalg.norm(x_MAP-x_true)/np.linalg.norm(x_true))

# %%
# sampling using NUTS
MCMC = cuqi.sampler.NUTS(TP.posterior)
Ns = int(200)      # number of samples
Nb = int(0.2*Ns)   # burn-in
samples = MCMC.sample(Ns, Nb)
#
xs = samples.samples
x_mean = np.mean(xs, axis=1)
x_std = np.std(xs, axis=1)
print('relative error mean:', np.linalg.norm(x_mean-x_true)/np.linalg.norm(x_true))

# %%
# plots
samples.plot_ci(exact=x_true)
TP.model.domain_geometry.plot(x_MAP, 'b-', label='MAP')
plt.legend(['Mean', 'True', 'MAP', 'CI'])
plt.show()

# %%
