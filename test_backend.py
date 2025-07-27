#!/usr/bin/env python3
"""
Simple test script to verify backend functionality
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cuqi.backend as backend
import numpy as np

def test_backend():
    print("Testing CUQIpy backend abstraction...")
    
    # Test default (numpy)
    print("\n1. Testing default backend (numpy):")
    backend.set("numpy")
    arr1 = backend.array([1, 2, 3, 4])
    print(f"   Created array: {arr1}")
    print(f"   Type: {type(arr1)}")
    print(f"   Shape: {arr1.shape}")
    
    # Test zeros
    zeros = backend.zeros((3, 4))
    print(f"   Zeros: {zeros}")
    print(f"   Type: {type(zeros)}")
    
    # Test array creation
    arr2 = backend.asarray([5, 6, 7, 8])
    print(f"   Asarray: {arr2}")
    print(f"   Type: {type(arr2)}")
    
    # Test is_array
    print(f"   Is array: {backend.is_array(arr1)}")
    print(f"   Is array (list): {backend.is_array([1, 2, 3])}")
    
    # Test PyTorch backend (if available)
    try:
        print("\n2. Testing PyTorch backend:")
        backend.set("torch")
        arr3 = backend.array([1, 2, 3, 4])
        print(f"   Created array: {arr3}")
        print(f"   Type: {type(arr3)}")
        print(f"   Shape: {arr3.shape}")
        
        # Test zeros
        zeros_torch = backend.zeros((3, 4))
        print(f"   Zeros: {zeros_torch}")
        print(f"   Type: {type(zeros_torch)}")
        
        # Test is_array
        print(f"   Is array: {backend.is_array(arr3)}")
        
        print("\n✅ PyTorch backend test passed!")
        
    except ImportError as e:
        print(f"\n⚠️  PyTorch not available: {e}")
        print("   Skipping PyTorch backend test")
    
    # Test CUQIarray
    print("\n3. Testing CUQIarray:")
    from cuqi.array import CUQIarray
    
    # Create CUQIarray with numpy backend
    backend.set("numpy")
    cuqi_arr = CUQIarray([1, 2, 3, 4])
    print(f"   CUQIarray: {cuqi_arr}")
    print(f"   Type: {type(cuqi_arr)}")
    print(f"   Shape: {cuqi_arr.shape}")
    print(f"   Array: {cuqi_arr.array}")
    
    # Test arithmetic operations
    result = cuqi_arr + cuqi_arr
    print(f"   Addition: {result}")
    print(f"   Type: {type(result)}")
    
    # Test matrix operations
    result2 = cuqi_arr @ cuqi_arr
    print(f"   Matrix multiply: {result2}")
    print(f"   Type: {type(result2)}")
    
    print("\n✅ Backend abstraction test completed!")

if __name__ == "__main__":
    test_backend() 