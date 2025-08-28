#!/usr/bin/env python3
"""
Test script for the refactored data collection module.
"""

import sys
import os

# Add the current directory to the path so we can import the modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_collection import (
    collect_variational_cooling_data,
    compute_ground_state_observables,
    create_system_observables
)

def test_observable_creation():
    """Test that system observables are created correctly."""
    print("Testing observable creation...")
    
    system_qubits = 4
    bath_qubits = 2
    
    observables, labels, types = create_system_observables(system_qubits, bath_qubits)
    
    print(f"Created {len(observables)} observables:")
    print(f"  Labels: {labels}")
    print(f"  Types: {types}")
    
    # Check that we have the right number of each type
    expected_X = system_qubits  # One X per system qubit
    expected_Z = system_qubits  # One Z per system qubit
    expected_ZZ = system_qubits * (system_qubits - 1) // 2  # Upper triangular ZZ pairs
    
    actual_X = sum(1 for t in types if t == "single_X")
    actual_Z = sum(1 for t in types if t == "single_Z")
    actual_ZZ = sum(1 for t in types if t == "correlation")
    
    print(f"Expected: {expected_X} X, {expected_Z} Z, {expected_ZZ} ZZ")
    print(f"Actual:   {actual_X} X, {actual_Z} Z, {actual_ZZ} ZZ")
    
    assert actual_X == expected_X, f"Expected {expected_X} X observables, got {actual_X}"
    assert actual_Z == expected_Z, f"Expected {expected_Z} Z observables, got {actual_Z}"
    assert actual_ZZ == expected_ZZ, f"Expected {expected_ZZ} ZZ observables, got {actual_ZZ}"
    
    print("✓ Observable creation test passed!")
    return True


def test_ground_state_computation():
    """Test ground state observable computation."""
    print("\nTesting ground state computation...")
    
    try:
        # Test with a small system
        system_qubits = 6
        J = 0.4
        h = 0.6
        
        ground_state_data = compute_ground_state_observables(
            system_qubits=system_qubits,
            J=J,
            h=h,
            bond_dim=20,  # Use smaller bond dimension for testing
            max_iter=50,   # Use fewer iterations for testing
            output_dir='test_results'
        )
        
        if 'error' in ground_state_data:
            print(f"✗ Ground state computation failed: {ground_state_data['error']}")
            return False
        
        print(f"Ground state energy: {ground_state_data['ground_state_results']['ground_state_energy']:.6f}")
        print(f"Number of observables: {len(ground_state_data['ground_state_results']['observables'])}")
        
        # Check that we have observables
        observables = ground_state_data['ground_state_results']['observables']
        if len(observables) > 0:
            print("✓ Ground state computation test passed!")
            return True
        else:
            print("✗ No observables computed")
            return False
            
    except Exception as e:
        print(f"✗ Ground state computation test failed: {e}")
        return False


def test_variational_cooling_data_collection():
    """Test variational cooling data collection."""
    print("\nTesting variational cooling data collection...")
    
    try:
        # Test with minimal parameters
        params = {
            'system_qubits': 4,
            'bath_qubits': 2,
            'open_boundary': 1,
            'J': 0.4,
            'h': 0.6,
            'p': 3,
            'num_sweeps': 2,
            'single_qubit_gate_noise': 0.0,  # No noise for testing
            'two_qubit_gate_noise': 0.0,
            'training_method': 'energy',
            'initial_state': 'zeros',
            'bond_dimensions': [8],  # Use small bond dimension for testing
            'num_shots': 10,         # Use few shots for testing
            'output_dir': 'test_results'
        }
        
        print("Running variational cooling simulation...")
        collected_data = collect_variational_cooling_data(**params)
        
        print(f"Data collection completed successfully!")
        print(f"Results saved to: {collected_data.get('metadata', {}).get('collection_timestamp', 'unknown')}")
        
        # Check that we have results
        if 'results' in collected_data and len(collected_data['results']) > 0:
            print("✓ Variational cooling data collection test passed!")
            return True
        else:
            print("✗ No results collected")
            return False
            
    except Exception as e:
        print(f"✗ Variational cooling data collection test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Refactored Data Collection Module")
    print("=" * 60)
    
    tests = [
        test_observable_creation,
        test_ground_state_computation,
        test_variational_cooling_data_collection
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! The refactored module is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    exit(main())
