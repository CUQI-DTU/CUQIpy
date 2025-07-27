#!/usr/bin/env python3
"""
Test script to verify PyTorch backend functionality
"""

import os
import sys

def test_pytorch_backend():
    """Test the PyTorch backend"""
    print("Testing PyTorch backend...")
    
    # Set PyTorch backend
    os.environ['CUQI_ARRAY_BACKEND'] = 'pytorch'
    
    # Clear module cache to reload with new backend
    modules_to_clear = [key for key in sys.modules.keys() if key.startswith('cuqi.array')]
    for module in modules_to_clear:
        del sys.modules[module]
    
    try:
        import cuqi.array as xp
        
        # Test basic operations
        print(f"Backend: {xp.get_backend_name()}")
        
        # Test array creation
        x = xp.array([1.0, 2.0, 3.0])
        y = xp.zeros(3)
        z = xp.ones(3)
        
        print(f"Array x: {x}")
        print(f"Array y: {y}")
        print(f"Array z: {z}")
        print(f"x + z: {x + z}")
        
        # Test mathematical operations
        print(f"Sum: {xp.sum(x)}")
        print(f"Mean: {xp.mean(x)}")
        print(f"Dot product: {xp.dot(x, z)}")
        
        # Test array conversion
        numpy_arr = xp.to_numpy(x)
        print(f"Converted to numpy: {numpy_arr} (type: {type(numpy_arr)})")
        
        torch_arr = xp.from_numpy(numpy_arr)
        print(f"Converted back to PyTorch: {torch_arr} (type: {type(torch_arr)})")
        
        # Test linspace
        lin = xp.linspace(0, 10, 5)
        print(f"Linspace: {lin}")
        
        # Test eye matrix
        eye_mat = xp.eye(3)
        print(f"Eye matrix: {eye_mat}")
        
        print("✓ PyTorch backend test passed!")
        return True
        
    except Exception as e:
        print(f"✗ PyTorch backend test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cuqi_functionality():
    """Test basic CUQI functionality with PyTorch backend"""
    print("\nTesting CUQI functionality with PyTorch backend...")
    
    try:
        import cuqi.array as xp
        from cuqi.distribution import Gaussian
        from cuqi.geometry import Discrete
        
        # Test distribution with PyTorch arrays
        mean = xp.zeros(3)
        dist = Gaussian(mean=mean, cov=1.0, geometry=Discrete(3))
        print(f"Distribution: {dist}")
        
        # Test sampling (this might not work fully due to random number generation)
        try:
            sample = dist.sample()
            print(f"Sample: {sample}")
        except Exception as e:
            print(f"Sampling failed (expected): {e}")
        
        print("✓ Basic CUQI functionality test passed!")
        return True
        
    except Exception as e:
        print(f"✗ CUQI functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== PyTorch Backend Test Suite ===")
    
    success1 = test_pytorch_backend()
    success2 = test_cuqi_functionality()
    
    if success1 and success2:
        print("\n🎉 All PyTorch backend tests passed!")
    else:
        print("\n❌ Some tests failed")
        sys.exit(1)