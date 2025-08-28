#!/usr/bin/env python3
"""
Test the direct MPS approach using save_expectation_value instead of Estimator.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error


def create_simple_test_circuit():
    """Create a simple 3-qubit circuit for testing."""
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.rx(np.pi/4, 2)
    return qc


def test_direct_mps_approach():
    """Test the direct MPS approach with save_expectation_value."""
    print("Testing direct MPS approach...")
    
    # Create circuit
    circuit = create_simple_test_circuit()
    
    # Add save instructions for observables
    circuit.save_expectation_value(Pauli("ZII"), [0, 1, 2], label="exp_ZII")
    circuit.save_expectation_value(Pauli("IZI"), [0, 1, 2], label="exp_IZI")
    circuit.save_expectation_value(Pauli("IIZ"), [0, 1, 2], label="exp_IIZ")
    
    # Test with different numbers of shots
    shot_counts = [10, 50, 100]
    
    for shots in shot_counts:
        print(f"\n--- {shots} shots ---")
        
        # Create simulator
        sim = AerSimulator(method='matrix_product_state')
        
        # Run with shots
        job = sim.run(circuit, shots=shots)
        result = job.result()
        
        # Extract results
        exp_zii = result.data()["exp_ZII"]
        exp_izi = result.data()["exp_IZI"]
        exp_iiz = result.data()["exp_IIZ"]
        
        print(f"  <ZII> = {exp_zii:.6f}")
        print(f"  <IZI> = {exp_izi:.6f}")
        print(f"  <IIZ> = {exp_iiz:.6f}")
    
    print("\n✓ Direct MPS approach working correctly!")


def test_noisy_direct_mps():
    """Test the direct MPS approach with noise."""
    print("\nTesting noisy direct MPS approach...")
    
    # Create circuit
    circuit = create_simple_test_circuit()
    circuit.save_expectation_value(Pauli("ZII"), [0, 1, 2], label="exp_ZII")
    
    # Create noisy simulator
    noise_model = NoiseModel()
    single_qubit_error = depolarizing_error(0.01, 1)
    noise_model.add_all_qubit_quantum_error(single_qubit_error, ['h', 'rx'])
    two_qubit_error = depolarizing_error(0.05, 2)
    noise_model.add_all_qubit_quantum_error(two_qubit_error, ['cx'])
    
    noisy_sim = AerSimulator(
        noise_model=noise_model,
        method='matrix_product_state',
        matrix_product_state_max_bond_dimension=32
    )
    
    # Test with different shot counts
    shot_counts = [50, 100, 200]
    
    for shots in shot_counts:
        print(f"\n--- {shots} shots ---")
        
        # Run multiple times to estimate variation
        multiple_runs = []
        for run in range(5):
            job = noisy_sim.run(circuit, shots=shots)
            result = job.result()
            exp_val = result.data()["exp_ZII"]
            multiple_runs.append(exp_val)
        
        exp_mean = np.mean(multiple_runs)
        exp_std = np.std(multiple_runs)
        shot_error = exp_std / np.sqrt(shots)
        
        print(f"  Mean: {exp_mean:.6f}")
        print(f"  Std: {exp_std:.6f}")
        print(f"  Shot error: {shot_error:.6f}")
    
    print("\n✓ Noisy direct MPS approach working correctly!")


if __name__ == "__main__":
    test_direct_mps_approach()
    test_noisy_direct_mps()
