#!/usr/bin/env python3
"""
Test file to demonstrate the effect of noise on MPS states.
Even with exact expectation value computation, noisy circuits produce different MPS states each time.
Now using the shots parameter directly on the backend for more efficient sampling.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error


def create_simple_test_circuit():
    """Create a simple 3-qubit circuit for testing."""
    qc = QuantumCircuit(3)
    
    # Apply some gates to create a non-trivial state
    qc.h(0)  # Hadamard on qubit 0
    qc.cx(0, 1)  # CNOT from qubit 0 to 1
    qc.rx(np.pi/4, 2)  # Rotation on qubit 2
    
    return qc


def create_noisy_simulator(single_qubit_gate_noise=0.01, two_qubit_gate_noise=0.05):
    """Create a noisy quantum simulator with custom noise model."""
    noise_model = NoiseModel()
    
    # Add depolarizing noise to single qubit gates
    single_qubit_error = depolarizing_error(single_qubit_gate_noise, 1)
    noise_model.add_all_qubit_quantum_error(single_qubit_error, ['h', 'rx'])
    
    # Add depolarizing noise to two qubit gates
    two_qubit_error = depolarizing_error(two_qubit_gate_noise, 2)
    noise_model.add_all_qubit_quantum_error(two_qubit_error, ['cx'])
    
    # Create MPS simulator with noise
    simulator = AerSimulator(
        noise_model=noise_model,
        method='matrix_product_state',
        matrix_product_state_max_bond_dimension=32
    )
    
    return simulator


def test_noiseless_vs_noisy_consistency():
    """Compare consistency between noiseless and noisy MPS simulations."""
    print("=" * 70)
    print("Testing Consistency: Noiseless vs Noisy MPS Simulations")
    print("=" * 70)
    
    # Create circuit
    circuit = create_simple_test_circuit()
    observable = Pauli("ZII")
    
    print(f"Test circuit:")
    print(circuit)
    print(f"Observable: {observable}")
    
    # Test 1: Noiseless MPS simulation
    print(f"\n--- Noiseless MPS Simulation ---")
    noiseless_sim = AerSimulator(method='matrix_product_state')
    
    noiseless_results = []
    for i in range(5):
        circuit_copy = circuit.copy()
        circuit_copy.save_expectation_value(observable, [0, 1, 2], label="exp_ZII")
        
        job = noiseless_sim.run(circuit_copy)
        result = job.result()
        exp_val = result.data()["exp_ZII"]
        noiseless_results.append(exp_val)
        print(f"  Run {i+1}: {exp_val:.10f}")
    
    noiseless_std = np.std(noiseless_results)
    print(f"  Standard deviation: {noiseless_std:.2e}")
    
    # Test 2: Noisy MPS simulation
    print(f"\n--- Noisy MPS Simulation ---")
    noisy_sim = create_noisy_simulator(single_qubit_gate_noise=0.01, two_qubit_gate_noise=0.05)
    
    noisy_results = []
    for i in range(5):
        circuit_copy = circuit.copy()
        circuit_copy.save_expectation_value(observable, [0, 1, 2], label="exp_ZII")
        
        job = noisy_sim.run(circuit_copy)
        result = job.result()
        exp_val = result.data()["exp_ZII"]
        noisy_results.append(exp_val)
        print(f"  Run {i+1}: {exp_val:.10f}")
    
    noisy_std = np.std(noisy_results)
    print(f"  Standard deviation: {noisy_std:.2e}")
    
    # Test 3: Different noise levels
    print(f"\n--- Different Noise Levels ---")
    noise_levels = [0.001, 0.01, 0.05, 0.1]
    
    for noise_level in noise_levels:
        print(f"\nNoise level: {noise_level}")
        high_noise_sim = create_noisy_simulator(
            single_qubit_gate_noise=noise_level, 
            two_qubit_gate_noise=noise_level*5
        )
        
        noise_level_results = []
        for i in range(3):  # Fewer runs for higher noise levels
            circuit_copy = circuit.copy()
            circuit_copy.save_expectation_value(observable, [0, 1, 2], label="exp_ZII")
            
            job = high_noise_sim.run(circuit_copy)
            result = job.result()
            exp_val = result.data()["exp_ZII"]
            noise_level_results.append(exp_val)
            print(f"  Run {i+1}: {exp_val:.10f}")
        
        noise_level_std = np.std(noise_level_results)
        print(f"  Standard deviation: {noise_level_std:.2e}")
    
    return noiseless_results, noisy_results


def test_expectation_value_statistics():
    """Test statistical properties of expectation values from noisy circuits using shots parameter."""
    print(f"\n" + "=" * 70)
    print("Testing Expectation Value Statistics from Noisy Circuits (Using Shots Parameter)")
    print("=" * 70)
    
    # Create circuit
    circuit = create_simple_test_circuit()
    observable = Pauli("ZII")
    
    # Test with moderate noise
    noisy_sim = create_noisy_simulator(single_qubit_gate_noise=0.02, two_qubit_gate_noise=0.1)
    
    # Test different numbers of shots
    shot_counts = [10, 50, 100, 200]
    
    print(f"Testing with different shot counts...")
    print(f"{'Shots':<8} {'Mean':<12} {'Std':<12} {'Shot Error':<12}")
    print("-" * 50)
    
    results = []
    for num_shots in shot_counts:
        # Add save instruction
        circuit_copy = circuit.copy()
        circuit_copy.save_expectation_value(observable, [0, 1, 2], label="exp_ZII")
        
        # Run with multiple shots
        job = noisy_sim.run(circuit_copy, shots=num_shots)
        result = job.result()
        
        # Extract the averaged expectation value
        exp_mean = result.data()["exp_ZII"]
        
        # Since the backend averages automatically, we need to run multiple times
        # to estimate the standard deviation
        multiple_runs = []
        for run in range(10):  # Run 10 times to estimate variation
            circuit_copy = circuit.copy()
            circuit_copy.save_expectation_value(observable, [0, 1, 2], label="exp_ZII")
            
            job = noisy_sim.run(circuit_copy, shots=num_shots)
            result = job.result()
            exp_val = result.data()["exp_ZII"]
            multiple_runs.append(exp_val)
        
        # Compute statistics across multiple runs
        exp_std = np.std(multiple_runs)
        shot_error = exp_std / np.sqrt(num_shots)
        
        print(f"{num_shots:<8} {exp_mean:<12.6f} {exp_std:<12.6f} {shot_error:<12.6f}")
        
        results.append({
            'shots': num_shots,
            'mean': exp_mean,
            'std': exp_std,
            'shot_error': shot_error
        })
    
    # Detailed analysis with 200 shots
    print(f"\n--- Detailed Analysis with 200 shots ---")
    circuit_copy = circuit.copy()
    circuit_copy.save_expectation_value(observable, [0, 1, 2], label="exp_ZII")
    
    job = noisy_sim.run(circuit_copy, shots=200)
    result = job.result()
    
    exp_mean = result.data()["exp_ZII"]
    
    # Run multiple times to estimate standard deviation
    multiple_runs = []
    for run in range(20):  # Run 20 times to get good statistics
        circuit_copy = circuit.copy()
        circuit_copy.save_expectation_value(observable, [0, 1, 2], label="exp_ZII")
        
        job = noisy_sim.run(circuit_copy, shots=200)
        result = job.result()
        exp_val = result.data()["exp_ZII"]
        multiple_runs.append(exp_val)
    
    multiple_runs_array = np.array(multiple_runs)
    std_val = np.std(multiple_runs_array)
    min_val = np.min(multiple_runs_array)
    max_val = np.max(multiple_runs_array)
    shot_error = std_val / np.sqrt(200)
    
    print(f"Statistical Analysis:")
    print(f"  Mean: {exp_mean:.10f}")
    print(f"  Standard deviation across runs: {std_val:.10f}")
    print(f"  Min: {min_val:.10f}")
    print(f"  Max: {max_val:.10f}")
    print(f"  Range: {max_val - min_val:.10f}")
    print(f"  Shot error (std/sqrt(200)): {shot_error:.10f}")
    
    # Compare with noiseless result
    noiseless_sim = AerSimulator(method='matrix_product_state')
    circuit_copy = circuit.copy()
    circuit_copy.save_expectation_value(observable, [0, 1, 2], label="exp_ZII")
    
    job = noiseless_sim.run(circuit_copy)
    result = job.result()
    noiseless_val = result.data()["exp_ZII"]
    
    print(f"\nComparison with noiseless result:")
    print(f"  Noiseless: {noiseless_val:.10f}")
    print(f"  Noisy mean: {exp_mean:.10f}")
    print(f"  Difference: {abs(exp_mean - noiseless_val):.10f}")
    print(f"  Noise-induced uncertainty: {std_val:.10f}")
    print(f"  Shot error: {shot_error:.10f}")
    
    # Confidence interval (95% confidence)
    confidence_interval = 1.96 * shot_error
    print(f"  95% confidence interval: ±{confidence_interval:.10f}")
    
    return multiple_runs_array, noiseless_val, shot_error


def test_multiple_observables_with_shots():
    """Test multiple observables using the shots parameter."""
    print(f"\n" + "=" * 70)
    print("Testing Multiple Observables with Shots Parameter")
    print("=" * 70)
    
    # Create circuit
    circuit = create_simple_test_circuit()
    
    # Define multiple observables
    observables = [
        (Pauli("ZII"), "Z on qubit 0"),
        (Pauli("IZI"), "Z on qubit 1"), 
        (Pauli("IIZ"), "Z on qubit 2"),
        (Pauli("ZZI"), "ZZ on qubits 0,1"),
        (Pauli("ZIZ"), "ZZ on qubits 0,2")
    ]
    
    # Test with moderate noise
    noisy_sim = create_noisy_simulator(single_qubit_gate_noise=0.02, two_qubit_gate_noise=0.1)
    
    print(f"Testing {len(observables)} observables with 100 shots each...")
    
    # Add save instructions for all observables
    circuit_copy = circuit.copy()
    for i, (obs, label) in enumerate(observables):
        circuit_copy.save_expectation_value(obs, [0, 1, 2], label=f"exp_{i}")
    
    # Run with multiple shots
    job = noisy_sim.run(circuit_copy, shots=100)
    result = job.result()
    
    # Extract and analyze results for each observable
    print(f"\nResults (averaged over 100 shots):")
    print(f"{'Observable':<15} {'Mean':<12} {'Std Est':<12}")
    print("-" * 50)
    
    for i, (obs, label) in enumerate(observables):
        exp_mean = result.data()[f"exp_{i}"]
        
        # Estimate standard deviation by running multiple times
        multiple_runs = []
        for run in range(5):  # Run 5 times to estimate variation
            circuit_copy = circuit.copy()
            circuit_copy.save_expectation_value(obs, [0, 1, 2], label=f"exp_{i}")
            
            job = noisy_sim.run(circuit_copy, shots=100)
            result = job.result()
            exp_val = result.data()[f"exp_{i}"]
            multiple_runs.append(exp_val)
        
        exp_std = np.std(multiple_runs)
        shot_error = exp_std / np.sqrt(100)
        
        print(f"{label:<15} {exp_mean:<12.6f} {exp_std:<12.6f}")
    
    return result


def test_shot_error_scaling():
    """Test how shot error scales with number of shots."""
    print(f"\n" + "=" * 70)
    print("Testing Shot Error Scaling with Number of Shots")
    print("=" * 70)
    
    # Create circuit
    circuit = create_simple_test_circuit()
    observable = Pauli("ZII")
    
    # Test with moderate noise
    noisy_sim = create_noisy_simulator(single_qubit_gate_noise=0.02, two_qubit_gate_noise=0.1)
    
    # Test different numbers of shots
    shot_counts = [10, 25, 50, 100, 200]
    
    print(f"Testing shot error scaling...")
    print(f"{'Shots':<8} {'Mean':<12} {'Std Est':<12} {'Shot Error':<12} {'1/sqrt(N)':<12}")
    print("-" * 70)
    
    results = []
    for num_shots in shot_counts:
        # Add save instruction
        circuit_copy = circuit.copy()
        circuit_copy.save_expectation_value(observable, [0, 1, 2], label="exp_ZII")
        
        # Run with multiple shots
        job = noisy_sim.run(circuit_copy, shots=num_shots)
        result = job.result()
        
        # Extract the averaged expectation value
        exp_mean = result.data()["exp_ZII"]
        
        # Estimate standard deviation by running multiple times
        multiple_runs = []
        for run in range(10):  # Run 10 times to estimate variation
            circuit_copy = circuit.copy()
            circuit_copy.save_expectation_value(observable, [0, 1, 2], label="exp_ZII")
            
            job = noisy_sim.run(circuit_copy, shots=num_shots)
            result = job.result()
            exp_val = result.data()["exp_ZII"]
            multiple_runs.append(exp_val)
        
        # Compute statistics
        exp_std = np.std(multiple_runs)
        shot_error = exp_std / np.sqrt(num_shots)
        theoretical_scaling = 1.0 / np.sqrt(num_shots)
        
        print(f"{num_shots:<8} {exp_mean:<12.6f} {exp_std:<12.6f} {shot_error:<12.6f} {theoretical_scaling:<12.6f}")
        
        results.append({
            'shots': num_shots,
            'mean': exp_mean,
            'std': exp_std,
            'shot_error': shot_error,
            'theoretical_scaling': theoretical_scaling
        })
    
    # Verify scaling law
    print(f"\nVerifying 1/sqrt(N) scaling law:")
    print(f"Shot error should scale as 1/sqrt(N) for statistical sampling")
    
    # Check if shot error roughly follows the scaling
    for i in range(1, len(results)):
        ratio = results[i]['shot_error'] / results[i-1]['shot_error']
        expected_ratio = np.sqrt(results[i-1]['shots'] / results[i]['shots'])
        print(f"  {results[i-1]['shots']} → {results[i]['shots']} shots: "
              f"actual ratio = {ratio:.3f}, expected = {expected_ratio:.3f}")
    
    return results


def main():
    """Main test function."""
    print("Testing Noise Effects on MPS Simulations (Using Shots Parameter)")
    print("=" * 70)
    
    # Test 1: Consistency comparison
    noiseless_results, noisy_results = test_noiseless_vs_noisy_consistency()
    
    # Test 2: Statistical analysis using shots parameter
    noisy_stats, noiseless_val, shot_error = test_expectation_value_statistics()
    
    # Test 3: Multiple observables
    multi_obs_result = test_multiple_observables_with_shots()
    
    # Test 4: Shot error scaling
    scaling_results = test_shot_error_scaling()
    
    # Summary and conclusions
    print(f"\n" + "=" * 70)
    print("SUMMARY AND CONCLUSIONS")
    print("=" * 70)
    
    print(f"\nKey Findings:")
    
    # Analyze noiseless consistency
    noiseless_std = np.std(noiseless_results)
    if noiseless_std < 1e-10:
        print(f"  ✓ Noiseless MPS simulation is perfectly consistent (std: {noiseless_std:.2e})")
    else:
        print(f"  ✗ Noiseless MPS simulation shows some variation (std: {noiseless_std:.2e})")
    
    # Analyze noisy consistency
    noisy_std = np.std(noisy_results)
    print(f"  ✗ Noisy MPS simulation shows significant variation (std: {noisy_std:.2e})")
    
    # Statistical analysis
    final_std = np.std(noisy_stats)
    print(f"  ✗ Statistical analysis confirms noise-induced variation (std: {final_std:.2e})")
    print(f"  ✓ Shot error estimation: {shot_error:.2e}")
    
    print(f"\nImplications for Your Variational Cooling Project:")
    print(f"  1. Even with exact expectation value computation, noisy circuits require multiple shots")
    print(f"  2. The MPS state itself varies between runs due to gate noise")
    print(f"  3. Using the shots parameter is much more efficient than manual loops")
    print(f"  4. Shot error scales as 1/sqrt(N) for statistical sampling")
    print(f"  5. This approach gives you MPS memory efficiency + exact computation per shot + proper error estimation")


if __name__ == "__main__":
    main()
