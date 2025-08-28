#!/usr/bin/env python3
"""
Test file to try the suggested MPS methods:
1. save_matrix_product_state - to get actual MPS tensors
2. save_expectation_value - to compute expectation values without sampling
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Pauli, Operator
from qiskit_aer import AerSimulator


def create_simple_test_circuit():
    """Create a simple 3-qubit circuit for testing."""
    qc = QuantumCircuit(3)
    
    # Apply some gates to create a non-trivial state
    qc.h(0)  # Hadamard on qubit 0
    qc.cx(0, 1)  # CNOT from qubit 0 to 1
    qc.rx(np.pi/4, 2)  # Rotation on qubit 2
    
    return qc


def test_save_matrix_product_state():
    """Test the save_matrix_product_state method to get actual MPS tensors."""
    print("=" * 60)
    print("Testing save_matrix_product_state method")
    print("=" * 60)
    
    # Create circuit
    qc = create_simple_test_circuit()
    print(f"Circuit:")
    print(qc)
    
    # Add save_matrix_product_state instruction
    qc.save_matrix_product_state(label="mps_state")
    
    # Use MPS simulator
    sim = AerSimulator(method="matrix_product_state")
    
    try:
        result = sim.run(qc).result()
        print(f"\nResult type: {type(result)}")
        print(f"Result data keys: {list(result.data(0).keys())}")
        
        if 'mps_state' in result.data(0):
            mps = result.data(0)['mps_state']
            print(f"\nMPS data type: {type(mps)}")
            print(f"MPS data shape: {np.array(mps).shape if hasattr(mps, 'shape') else 'No shape attribute'}")
            print(f"MPS data length: {len(mps) if hasattr(mps, '__len__') else 'No length attribute'}")
            
            # Try to understand the structure
            if isinstance(mps, list):
                print(f"\nMPS is a list with {len(mps)} elements")
                for i, tensor in enumerate(mps):
                    print(f"  Site {i}: type={type(tensor)}, shape={np.array(tensor).shape if hasattr(tensor, 'shape') else 'No shape'}")
                    if i < 2:  # Show first few tensors
                        print(f"    Tensor data: {tensor}")
            
            return mps
        else:
            print(f"\nNo 'mps_state' found in result data")
            print(f"Available data: {result.data(0)}")
            return None
            
    except Exception as e:
        print(f"Error with save_matrix_product_state: {e}")
        return None


def test_save_expectation_value():
    """Test the save_expectation_value method to compute expectation values without sampling."""
    print(f"\n" + "=" * 60)
    print("Testing save_expectation_value method")
    print("=" * 60)
    
    # Create circuit
    qc = create_simple_test_circuit()
    
    # Add save_expectation_value instructions for different observables
    qc.save_expectation_value(Pauli("ZII"), [0, 1, 2], label="exp_ZII")
    qc.save_expectation_value(Pauli("IZI"), [0, 1, 2], label="exp_IZI")
    qc.save_expectation_value(Pauli("IIZ"), [0, 1, 2], label="exp_IIZ")
    qc.save_expectation_value(Pauli("ZZI"), [0, 1, 2], label="exp_ZZI")
    
    print(f"Circuit with expectation value saves:")
    print(qc)
    
    # Use MPS simulator
    sim = AerSimulator(method="matrix_product_state")
    
    try:
        result = sim.run(qc).result()
        print(f"\nResult type: {type(result)}")
        print(f"Result data keys: {list(result.data(0).keys())}")
        
        # Extract expectation values
        expectations = {}
        for label in ["exp_ZII", "exp_IZI", "exp_IIZ", "exp_ZZI"]:
            if label in result.data(0):
                expectations[label] = result.data(0)[label]
                print(f"{label}: {expectations[label]}")
            else:
                print(f"{label}: Not found")
        
        return expectations
        
    except Exception as e:
        print(f"Error with save_expectation_value: {e}")
        return None


def test_save_expectation_value_variance():
    """Test the save_expectation_value_variance method."""
    print(f"\n" + "=" * 60)
    print("Testing save_expectation_value_variance method")
    print("=" * 60)
    
    # Create circuit
    qc = create_simple_test_circuit()
    
    # Add save_expectation_value_variance instruction
    qc.save_expectation_value_variance(Pauli("ZII"), [0, 1, 2], label="expvar_ZII")
    
    print(f"Circuit with expectation value and variance save:")
    print(qc)
    
    # Use MPS simulator
    sim = AerSimulator(method="matrix_product_state")
    
    try:
        result = sim.run(qc).result()
        print(f"\nResult type: {type(result)}")
        print(f"Result data keys: {list(result.data(0).keys())}")
        
        if 'expvar_ZII' in result.data(0):
            exp_var = result.data(0)['expvar_ZII']
            print(f"expvar_ZII: {exp_var}")
            print(f"Type: {type(exp_var)}")
            if hasattr(exp_var, '__len__'):
                print(f"Length: {len(exp_var)}")
                print(f"Expectation: {exp_var[0]}")
                print(f"Variance: {exp_var[1]}")
            return exp_var
        else:
            print(f"expvar_ZII not found in result data")
            return None
            
    except Exception as e:
        print(f"Error with save_expectation_value_variance: {e}")
        return None


def test_arbitrary_observable():
    """Test with an arbitrary Hermitian observable."""
    print(f"\n" + "=" * 60)
    print("Testing arbitrary Hermitian observable")
    print("=" * 60)
    
    # Create circuit
    qc = create_simple_test_circuit()
    
    # Create arbitrary observable: (X + Y)/sqrt(2) on qubit 0
    obs_matrix = (1/np.sqrt(2)) * (np.array([[0,1],[1,0]]) + np.array([[0,-1j],[1j,0]]))
    obs = Operator(obs_matrix)
    
    print(f"Observable matrix:")
    print(obs_matrix)
    
    # Add save_expectation_value for arbitrary observable
    qc.save_expectation_value(obs, [0], label="exp_arbitrary")
    
    print(f"Circuit with arbitrary observable save:")
    print(qc)
    
    # Use MPS simulator
    sim = AerSimulator(method="matrix_product_state")
    
    try:
        result = sim.run(qc).result()
        print(f"\nResult type: {type(result)}")
        print(f"Result data keys: {list(result.data(0).keys())}")
        
        if 'exp_arbitrary' in result.data(0):
            exp_val = result.data(0)['exp_arbitrary']
            print(f"exp_arbitrary: {exp_val}")
            return exp_val
        else:
            print(f"exp_arbitrary not found in result data")
            return None
            
    except Exception as e:
        print(f"Error with arbitrary observable: {e}")
        return None


def main():
    """Main test function."""
    print("Testing MPS Methods in Qiskit")
    print("=" * 60)
    
    # Test 1: save_matrix_product_state
    mps_result = test_save_matrix_product_state()
    
    # Test 2: save_expectation_value
    exp_result = test_save_expectation_value()
    
    # Test 3: save_expectation_value_variance
    expvar_result = test_save_expectation_value_variance()
    
    # Test 4: arbitrary observable
    arbitrary_result = test_arbitrary_observable()
    
    # Summary
    print(f"\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print(f"save_matrix_product_state: {'✓' if mps_result is not None else '✗'}")
    print(f"save_expectation_value: {'✓' if exp_result is not None else '✗'}")
    print(f"save_expectation_value_variance: {'✓' if expvar_result is not None else '✗'}")
    print(f"arbitrary observable: {'✓' if arbitrary_result is not None else '✗'}")
    
    if mps_result is not None:
        print(f"\nMPS tensors successfully extracted!")
        print(f"This gives you access to the actual MPS representation")
    
    if exp_result is not None:
        print(f"\nExpectation values computed without sampling!")
        print(f"This bypasses the Estimator's sampling behavior")
    
    if expvar_result is not None:
        print(f"\nExpectation values and variances computed!")
        print(f"This gives you both the mean and uncertainty")


if __name__ == "__main__":
    main()
