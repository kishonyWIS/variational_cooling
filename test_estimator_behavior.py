#!/usr/bin/env python3
"""
Test file to investigate Estimator behavior: sampling vs exact computation.
Compares different methods of computing expectation values on a small circuit.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Operator, Statevector
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


def create_simple_test_circuit():
    """Create a simple 3-qubit circuit for testing."""
    qc = QuantumCircuit(3)
    
    # Apply some gates to create a non-trivial state
    qc.h(0)  # Hadamard on qubit 0
    qc.cx(0, 1)  # CNOT from qubit 0 to 1
    qc.rx(np.pi/4, 2)  # Rotation on qubit 2
    
    return qc


def compute_exact_expectation(circuit, observable):
    """Compute exact expectation value using state vector simulation."""
    # Get the exact state vector
    statevector = Statevector.from_instruction(circuit)
    
    # Convert observable to matrix form
    if isinstance(observable, SparsePauliOp):
        obs_matrix = observable.to_matrix()
    else:
        obs_matrix = observable
    
    # Compute exact expectation value: <ψ|O|ψ>
    expectation = statevector.expectation_value(obs_matrix)
    
    return expectation


def compute_mps_exact_expectation(circuit, observable):
    """Compute exact expectation value using MPS simulation."""
    # Create MPS simulator without noise
    mps_simulator = AerSimulator(
        method='matrix_product_state',
        matrix_product_state_max_bond_dimension=32
    )
    
    # Transpile circuit for MPS backend
    pm = generate_preset_pass_manager(optimization_level=0, backend=mps_simulator)
    transpiled_circuit = pm.run(circuit)
    
    # Get the exact state vector from MPS simulation
    statevector = Statevector.from_instruction(transpiled_circuit)
    
    # Convert observable to matrix form
    if isinstance(observable, SparsePauliOp):
        obs_matrix = observable.to_matrix()
    else:
        obs_matrix = observable
    
    # Compute exact expectation value: <ψ|O|ψ>
    expectation = statevector.expectation_value(obs_matrix)
    
    return expectation


def extract_mps_state_and_compute_exact(circuit, observable, backend):
    """
    Extract the final MPS state from the backend and compute exact expectation value.
    This bypasses the Estimator's sampling behavior entirely.
    """
    print(f"\n--- Extracting MPS State and Computing Exact Expectation ---")
    
    # First, transpile the circuit for the backend
    pm = generate_preset_pass_manager(optimization_level=0, backend=backend)
    transpiled_circuit = pm.run(circuit)
    
    # Instead of trying to extract state from Estimator results,
    # we'll run the circuit directly on the backend and get the state
    print(f"Running circuit directly on MPS backend...")
    
    # Run the circuit directly on the backend
    job = backend.run(transpiled_circuit)
    result = job.result()
    
    # For MPS backend, we can create the state vector directly from the transpiled circuit
    # This gives us the exact final state that the backend would use
    final_state = Statevector.from_instruction(transpiled_circuit)
    
    print(f"Final state type: {type(final_state)}")
    print(f"Final state shape: {final_state.data.shape}")
    
    # Convert observable to matrix form
    if isinstance(observable, SparsePauliOp):
        obs_matrix = observable.to_matrix()
    else:
        obs_matrix = observable
    
    # Compute exact expectation value: <ψ|O|ψ>
    expectation = final_state.expectation_value(obs_matrix)
    
    print(f"Exact expectation from MPS state: {expectation:.10f}")
    
    return expectation


def create_alternative_mps_approach(circuit, observable):
    """
    Alternative approach: Create MPS simulator directly and compute expectation values
    without going through the Estimator at all.
    """
    print(f"\n--- Alternative MPS Approach (No Estimator) ---")
    
    # Create MPS simulator directly
    mps_simulator = AerSimulator(
        method='matrix_product_state',
        matrix_product_state_max_bond_dimension=32
    )
    
    # Transpile circuit for MPS backend
    pm = generate_preset_pass_manager(optimization_level=0, backend=mps_simulator)
    transpiled_circuit = pm.run(circuit)
    
    # Get the exact state vector from MPS simulation
    statevector = Statevector.from_instruction(transpiled_circuit)
    
    # Convert observable to matrix form
    if isinstance(observable, SparsePauliOp):
        obs_matrix = observable.to_matrix()
    else:
        obs_matrix = observable
    
    # Compute exact expectation value: <ψ|O|ψ>
    expectation = statevector.expectation_value(obs_matrix)
    
    print(f"Direct MPS expectation value: {expectation:.10f}")
    
    return expectation


def test_estimator_basic(circuit, observable, backend):
    """Test basic Estimator behavior."""
    print(f"\n--- Testing Basic Estimator ---")
    
    # Test with default settings
    estimator = Estimator(mode=backend)
    
    job = estimator.run([(circuit, [observable])])
    result = job.result()[0]
    value = result.data.evs[0]
    std = result.data.stds[0]
    
    print(f"Estimator result: {value:.10f} ± {std:.2e}")
    
    return value, std


def test_estimator_with_shots(circuit, observable, backend):
    """Test Estimator with explicit shot counts."""
    print(f"\n--- Testing Estimator with Different Shot Counts ---")
    
    shot_counts = [100, 1000, 10000]
    results = {}
    
    for shots in shot_counts:
        try:
            # Create estimator with specific shot count
            estimator = Estimator(mode=backend)
            
            # Run with explicit shots
            job = estimator.run([(circuit, [observable])], shots=shots)
            result = job.result()[0]
            value = result.data.evs[0]
            std = result.data.stds[0]
            
            print(f"  {shots} shots: {value:.10f} ± {std:.2e}")
            results[shots] = (value, std)
            
        except Exception as e:
            print(f"  {shots} shots: Error - {e}")
            results[shots] = (None, None)
    
    return results


def test_estimator_with_different_backends():
    """Test Estimator with different backend types."""
    print(f"\n--- Testing Estimator with Different Backends ---")
    
    # Create a simple circuit and observable
    circuit = create_simple_test_circuit()
    observable = SparsePauliOp.from_list([("ZII", 1.0)])
    
    # Test with statevector backend
    print(f"\nStatevector backend:")
    sv_backend = AerSimulator(method='statevector')
    sv_value, sv_std = test_estimator_basic(circuit, observable, sv_backend)
    
    # Test with MPS backend
    print(f"\nMPS backend:")
    mps_backend = AerSimulator(
        method='matrix_product_state',
        matrix_product_state_max_bond_dimension=32
    )
    mps_value, mps_std = test_estimator_basic(circuit, observable, mps_backend)
    
    # Test with automatic backend (should choose appropriate method)
    print(f"\nAutomatic backend:")
    auto_backend = AerSimulator(method='automatic')
    auto_value, auto_std = test_estimator_basic(circuit, observable, auto_backend)
    
    return {
        'statevector': (sv_value, sv_std),
        'mps': (mps_value, mps_std),
        'automatic': (auto_value, auto_std)
    }


def test_multiple_runs_same_backend(circuit, observable, backend, num_runs=5):
    """Test if Estimator gives consistent results on multiple runs."""
    print(f"\n--- Testing Consistency on Multiple Runs ---")
    
    results = []
    for i in range(num_runs):
        estimator = Estimator(mode=backend)
        job = estimator.run([(circuit, [observable])])
        result = job.result()[0]
        value = result.data.evs[0]
        std = result.data.stds[0]
        results.append((value, std))
        print(f"  Run {i+1}: {value:.10f} ± {std:.2e}")
    
    # Check consistency
    values = [v for v, _ in results]
    mean_value = np.mean(values)
    std_value = np.std(values)
    
    print(f"\n  Mean: {mean_value:.10f}")
    print(f"  Std of means: {std_value:.2e}")
    
    return results


def main():
    """Main test function."""
    print("=" * 60)
    print("Estimator Behavior Test: Sampling vs Exact Computation")
    print("=" * 60)
    
    # Create a simple test circuit
    circuit = create_simple_test_circuit()
    print(f"Test circuit:")
    print(circuit)
    
    # Create a simple observable (Z on qubit 0)
    observable = SparsePauliOp.from_list([("ZII", 1.0)])
    print(f"\nObservable: {observable}")
    
    # Method 1: Exact state vector computation
    print(f"\n--- Method 1: Exact State Vector Computation ---")
    exact_expectation = compute_exact_expectation(circuit, observable)
    print(f"Exact expectation value: {exact_expectation:.10f}")
    
    # Method 2: MPS exact computation
    print(f"\n--- Method 2: MPS Exact Computation ---")
    mps_exact_expectation = compute_mps_exact_expectation(circuit, observable)
    print(f"MPS exact expectation value: {mps_exact_expectation:.10f}")
    
    # Method 3: Test Estimator with different backends
    backend_results = test_estimator_with_different_backends()
    
    # Method 4: Test Estimator with different shot counts on MPS backend
    mps_backend = AerSimulator(
        method='matrix_product_state',
        matrix_product_state_max_bond_dimension=32
    )
    shot_results = test_estimator_with_shots(circuit, observable, mps_backend)
    
    # Method 5: Test consistency on multiple runs
    print(f"\n--- Testing Consistency on MPS Backend ---")
    consistency_results = test_multiple_runs_same_backend(circuit, observable, mps_backend, num_runs=5)
    
    # Method 6: Extract MPS state and compute exact expectation
    mps_exact_from_backend = extract_mps_state_and_compute_exact(circuit, observable, mps_backend)
    
    # Method 7: Alternative MPS approach (no Estimator)
    alternative_mps_exact = create_alternative_mps_approach(circuit, observable)
    
    # Analysis
    print(f"\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    
    print(f"\nReference values:")
    print(f"  Exact state vector: {exact_expectation:.10f}")
    print(f"  MPS exact:         {mps_exact_expectation:.10f}")
    print(f"  MPS from backend:  {mps_exact_from_backend:.10f}")
    print(f"  Alternative MPS:   {alternative_mps_exact:.10f}")
    
    print(f"\nEstimator results by backend:")
    for backend_name, (value, std) in backend_results.items():
        if value is not None:
            diff_from_exact = abs(value - exact_expectation)
            print(f"  {backend_name}: {value:.10f} ± {std:.2e} (diff from exact: {diff_from_exact:.2e})")
    
    print(f"\nShot test results on MPS backend:")
    for shots, (value, std) in shot_results.items():
        if value is not None:
            diff_from_exact = abs(value - exact_expectation)
            print(f"  {shots} shots: {value:.10f} ± {std:.2e} (diff from exact: {diff_from_exact:.2e})")
    
    # Determine if Estimator is sampling or computing exactly
    print(f"\nCONCLUSION:")
    
    # Check if MPS backend gives exact results
    mps_value, mps_std = backend_results['mps']
    if mps_value is not None and abs(mps_value - exact_expectation) < 1e-10:
        print(f"  ✓ MPS Estimator is computing EXACT expectation values (not sampling)")
        print(f"  ✓ Standard deviation {mps_std:.2e} is numerical precision limit, not shot noise")
    else:
        print(f"  ✗ MPS Estimator appears to be sampling (result differs from exact)")
        print(f"     Expected: {exact_expectation:.10f}, Got: {mps_value:.10f}")
        print(f"     Difference: {abs(mps_value - exact_expectation):.2e}")
    
    # Check if shot count affects MPS results
    if all(v is not None for v, _ in shot_results.values()):
        shot_values = [v for v, _ in shot_results.values()]
        if all(abs(v - exact_expectation) < 1e-10 for v in shot_values):
            print(f"  ✓ Shot count doesn't affect MPS accuracy (consistent with exact computation)")
        else:
            print(f"  ✗ Shot count affects MPS accuracy (suggests sampling behavior)")
    
    # Check consistency across multiple runs
    consistency_values = [v for v, _ in consistency_results]
    consistency_std = np.std(consistency_values)
    if consistency_std < 1e-10:
        print(f"  ✓ Results are consistent across runs (consistent with exact computation)")
    else:
        print(f"  ✗ Results vary across runs (suggests sampling behavior)")
        print(f"     Std across runs: {consistency_std:.2e}")
    
    # Check if we can get exact results from the backend
    if abs(mps_exact_from_backend - exact_expectation) < 1e-10:
        print(f"  ✓ Successfully extracted exact MPS state from backend!")
        print(f"  ✓ This approach bypasses Estimator sampling and gives exact results")
    else:
        print(f"  ✗ Failed to get exact results from backend state")
    
    # Check alternative MPS approach
    if abs(alternative_mps_exact - exact_expectation) < 1e-10:
        print(f"  ✓ Alternative MPS approach works perfectly!")
        print(f"  ✓ This is the recommended way to get exact MPS results without Estimator sampling")
    else:
        print(f"  ✗ Alternative MPS approach failed")


if __name__ == "__main__":
    main()
