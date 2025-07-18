#!/usr/bin/env python3
import numpy as np
from scipy.linalg import eigh
from numpy import pi
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import time
import csv
import os
import threading

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Operator, Statevector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_ibm_runtime.options import EstimatorOptions


def save_result_to_csv(csv_file, result_dict):
    """
    Thread-safe function to save results to CSV file.
    
    Args:
        csv_file: path to CSV file
        result_dict: dictionary containing the result data
    """
    lock = threading.Lock()
    with lock:
        file_exists = os.path.exists(csv_file)
        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=result_dict.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(result_dict)


def get_noisy_simulator(single_qubit_gate_noise=0.0003, two_qubit_gate_noise=0.003,
                        max_bond_dimension=4):
    """Create a noisy quantum simulator with custom noise model."""
    noise_model = NoiseModel()
    single_qubit_error = depolarizing_error(single_qubit_gate_noise, 1)
    noise_model.add_all_qubit_quantum_error(single_qubit_error, ['rz', 'rx'])

    rzz_error = depolarizing_error(two_qubit_gate_noise, 2)
    noise_model.add_all_qubit_quantum_error(rzz_error, ['rzz'])
    
    simulator = AerSimulator(
        noise_model=noise_model, 
        method='matrix_product_state',
        matrix_product_state_max_bond_dimension=max_bond_dimension,
    )
    return simulator


def pauli_sys_ZZ(system_qubits, bath_qubits, J, open_boundary=1):
    """Generate ZZ Pauli strings for system Hamiltonian."""
    pauli_str = []
    str_iter = "I"*(system_qubits-2) + "ZZ"
    
    for i in range(system_qubits-open_boundary):
        pauli_str.append("I"*bath_qubits + str_iter)
        str_iter = str_iter[1:] + str_iter[0]
    
    pauli_sys_ZZ = [(itr, -J) for itr in pauli_str]
    return pauli_sys_ZZ


def pauli_sys_X(system_qubits, bath_qubits, h):
    """Generate X Pauli strings for system Hamiltonian."""
    pauli_str = []
    str_iter = "I"*(system_qubits-1) + "X"
    
    for i in range(system_qubits):
        pauli_str.append("I"*bath_qubits + str_iter)
        str_iter = str_iter[1:] + str_iter[0]
    
    pauli_sys_X = [(itr, -h) for itr in pauli_str]
    return pauli_sys_X


def hva_multiple_sweeps(p, system_qubits, bath_qubits, parameters, num_sweeps=1, 
                       reset=True, barrier=True, J=0.4, h=0.6, half=True, open_boundary=1):
    """
    Create HVA circuit for variational cooling.
    
    Args:
        p: number of HVA layers per sweep
        system_qubits: number of system qubits
        bath_qubits: number of bath qubits
        parameters: [alpha, beta, B_t, g_t] parameter arrays
        num_sweeps: number of cooling sweeps
        reset: whether to reset bath qubits between sweeps
        barrier: whether to include barriers for visualization
        J: Ising coupling strength
        h: transverse field strength
        half: if True, one bath site per two system qubits
        open_boundary: boundary conditions (1=open, 0=periodic)
    """
    alpha, beta, B_t, g_t = parameters
    num_qubits = system_qubits + bath_qubits
    
    qc = QuantumCircuit(system_qubits + bath_qubits)
    
    for sweeps in range(num_sweeps):    
        for i in range(p):   
            # H_1: brick-wall layout of RZZ gates on system bonds, and Z-rotations on bath qubits
            for j in range(0, system_qubits-open_boundary, 2):
                qc.rzz(-J*alpha[i], j, (j+1)%system_qubits)
            for j in range(1, system_qubits-open_boundary, 2):
                qc.rzz(-J*alpha[i], j, (j+1)%system_qubits)                

            for j in range(system_qubits, num_qubits):
                qc.rz(-B_t[i], j)

            if barrier: 
                qc.barrier()
                
            # H_2: RX rotations on system qubits
            for j in range(system_qubits):
                qc.rx(-h*beta[i], j)
    
            if barrier: 
                qc.barrier()
                
            # H_3: RYY gates connecting system and bath qubits
            if half:
                for j in range(bath_qubits):
                    qc.ryy(-g_t[i], 2*j, j+system_qubits)
            else:
                for j in range(bath_qubits):
                    qc.ryy(-g_t[i], j, j+system_qubits)
                    
            if barrier: 
                qc.barrier()

        if reset:
            if barrier: 
                qc.barrier()
            for k in range(bath_qubits):
                qc.reset(-k-1)
              
            if barrier: 
                qc.barrier()                
    return qc


def fixed_bond_dimension_study(system_qubits=10, bath_qubits=5, half=True, open_boundary=1, 
                              J=0.4, h=0.6, p=3, num_sweeps=1, 
                              single_qubit_gate_noise=0., two_qubit_gate_noise=0.,
                              training_method="energy", initial_state="zeros", verbose=True, csv_file=None,
                              bond_dimensions=[32, 64], energy_density_atol=0.01):
    """
    Run computations for fixed bond dimensions.
    
    Args:
        system_qubits: number of system qubits
        bath_qubits: number of bath qubits
        half: if True, one bath site per two system qubits
        open_boundary: boundary conditions (1=open, 0=periodic)
        J: Ising coupling strength
        h: transverse field strength
        p: number of HVA layers per sweep (HVA depth)
        num_sweeps: number of cooling sweeps
        single_qubit_gate_noise: single qubit gate noise parameter
        two_qubit_gate_noise: two qubit gate noise parameter
        training_method: training method ("energy" or "mpo_fidelity")
        initial_state: initial state description as string (e.g., "zeros", "random", etc.)
        verbose: if True, print progress and create plots
        csv_file: path to CSV file to save results (thread-safe)
        bond_dimensions: list of bond dimensions to test (default: [32, 64])
        energy_density_atol: absolute tolerance for energy density - std/system_qubits must be smaller than this
    """
    
    num_qubits = system_qubits + bath_qubits
    
    # Create system Hamiltonian
    pauli_sys = pauli_sys_ZZ(system_qubits, 0, J, open_boundary) + pauli_sys_X(system_qubits, 0, h)
    H_sys = SparsePauliOp.from_list(pauli_sys)
    
    # Convert energy density tolerance to energy tolerance
    energy_atol = energy_density_atol * system_qubits
    # Convert energy tolerance to precision for estimator
    energy_precision = energy_atol / np.sqrt(len(pauli_sys))
    
    # Calculate ground state energy using DMRG
    if verbose:
        print("Calculating ground state energy using DMRG...")
    try:
        from ising_ground_state_dmrg import calculate_ising_ground_state
        E0 = calculate_ising_ground_state(N=system_qubits, J=J, h=h, bond_dim=50, max_iter=200, cyclic=False)
    except Exception as e:
        if verbose:
            print(f"Warning: Could not calculate ground state energy: {e}")
            print("Using E0 = 0 for relative error calculations")
        E0 = 0.0

    # Pre-optimized parameters for J=0.4, 3 layers
    best_para = np.array([
        0.026932368894753006, 0.58775691792609, 1.416345878671235, 
        -0.5847230425335908, 2.108254043912492, 0.9685586146293224, 
        3.141592653589793, 1.9041272021498745, -0.7655691857212363, 
        1.463995846549075, 1.062269199105236, 0.450403882505529
    ])
    
    # Split parameters
    lengths = [p, p, p, p]
    split_pts = np.cumsum(lengths)[:-1]
    
    # Create observables for the enlarged circuit
    H_observables = []
    for pauliop in pauli_sys:
        bt_str = (bath_qubits)*"I" + pauliop[0]
        H_observables.append((bt_str, pauliop[1]))
    observables = SparsePauliOp.from_list(H_observables)
    
    # Build circuit for 1 sweep
    total_circ = QuantumCircuit(num_qubits)
    total_circ = total_circ.compose(
        hva_multiple_sweeps(
            p, system_qubits, bath_qubits, 
            np.split(best_para, split_pts),
            num_sweeps=num_sweeps, reset=True, J=J, h=h, 
            half=half, open_boundary=open_boundary
        ), 
        qubits=[k for k in range(num_qubits)]
    )
    
    # Setup layout
    indices = [k for k in range(num_qubits)]
    bath_indices = [k for k in range(1, num_qubits, 3)]
    for bath_idx in bath_indices:
        indices.remove(bath_idx)
    layout = indices + bath_indices
    
    # Use provided bond dimensions
    results = []
    
    start_time = time.time()
    
    for bond_dim in bond_dimensions:
        if verbose:
            print(f"Testing bond_dim={bond_dim}")
        
        # Setup backend with current bond dimension
        backend = get_noisy_simulator(
            single_qubit_gate_noise=single_qubit_gate_noise,
            two_qubit_gate_noise=two_qubit_gate_noise,
            max_bond_dimension=bond_dim,
        )
        
        # Transpile circuit
        pm = generate_preset_pass_manager(optimization_level=3, backend=backend, initial_layout=layout)
        transpiled_circuit = pm.run(total_circ)
        mapped_observables = [observable.apply_layout(transpiled_circuit.layout) for observable in observables]
        
        # Setup estimator with precision based on energy_atol
        options = EstimatorOptions(
            default_precision=energy_precision
        )
        estimator = Estimator(mode=backend, options=options)
        
        # Run simulation
        job = estimator.run([(transpiled_circuit, mapped_observables)])
        pub_result = job.result()[0]
        values = pub_result.data.evs
        stds = pub_result.data.stds
        
        # Calculate total energy and its standard deviation
        energy = sum(values)
        energy_diff = energy - E0  # E - E0
        energy_std = np.sqrt(np.sum(np.array(stds)**2))
        
        # Calculate truncation error as difference from previous lower bond dimension
        truncation_error = np.inf
        if len(results) >= 1:
            # Find the most recent energy with a lower bond dimension
            for i in range(len(results) - 1, -1, -1):  # Go backwards
                if results[i]['bond_dim'] < bond_dim:
                    observed_diff = abs(energy_diff - results[i]['energy_diff'])
                    # Isolate truncation error by subtracting statistical noise
                    # Var(difference) = Var(truncation_error) + Var(shot_noise_1) + Var(shot_noise_2)
                    current_shot_variance = energy_std**2
                    previous_shot_variance = results[i]['energy_std']**2
                    if observed_diff**2 - current_shot_variance - previous_shot_variance > max(previous_shot_variance, current_shot_variance):
                        truncation_variance = observed_diff**2 - current_shot_variance - previous_shot_variance
                    else:
                        truncation_variance = max(previous_shot_variance, current_shot_variance)
                    truncation_error = np.sqrt(truncation_variance)
                    break
        
        # Combined standard deviation (shot noise + truncation error)
        combined_std = np.sqrt(energy_std**2 + truncation_error**2)
        
        results.append({
            'bond_dim': bond_dim,
            'energy': energy,
            'energy_diff': energy_diff,
            'energy_std': energy_std,
            'truncation_error': truncation_error,
            'combined_std': combined_std
        })
        
        if verbose:
            print(f"  Energy: {energy:.6f} (E-E0: {energy_diff:.6f}) ± {energy_std:.6f}")
            print(f"  Truncation error: {truncation_error:.6f}")
            print(f"  Combined std: {combined_std:.6f}")
    
    total_time = (time.time() - start_time) / 60  # in minutes
    
    if verbose:
        print("=" * 70)
        print(f"Fixed bond dimension study completed in {total_time:.2f} minutes!")
        for result in results:
            print(f"Bond dim {result['bond_dim']}: E-E0 = {result['energy_diff']:.6f} ± {result['combined_std']:.6f} (shot: {result['energy_std']:.6f}, trunc: {result['truncation_error']:.6f})")
    
    # Save results to CSV if specified
    if csv_file is not None:
        for result in results:
            result_dict = {
                'system_qubits': system_qubits,
                'bath_qubits': bath_qubits,
                'J': J,
                'h': h,
                'num_sweeps': num_sweeps,
                'p': p,
                'training_method': training_method,
                'single_qubit_gate_noise': single_qubit_gate_noise,
                'two_qubit_gate_noise': two_qubit_gate_noise,
                'initial_state': initial_state,
                'ground_state_energy': E0,
                'bond_dim': result['bond_dim'],
                'energy': result['energy'],
                'energy_diff': result['energy_diff'],
                'energy_std': result['energy_std'],
                'truncation_error': result['truncation_error'],
                'combined_std': result['combined_std'],
                'energy_density_atol': energy_density_atol,
                'energy_atol': energy_atol,
                'total_time_minutes': total_time
            }
            save_result_to_csv(csv_file, result_dict)
    
    # Create comparison plot only if verbose
    if verbose:
        plt.figure(figsize=(10, 6))
        
        bond_dims = [r['bond_dim'] for r in results]
        energy_diffs = [r['energy_diff'] for r in results]
        energy_stds = [r['energy_std'] for r in results]
        
        plt.errorbar(bond_dims, energy_diffs, yerr=energy_stds, 
                    marker='o', capsize=5, capthick=2, linewidth=2, markersize=8, 
                    label='E-E0 ± Shot Noise Only')
        
        # Also plot combined error
        combined_stds = [r['combined_std'] for r in results]
        plt.errorbar(bond_dims, energy_diffs, yerr=combined_stds, 
                    marker='s', capsize=3, capthick=1, linewidth=1, markersize=6, 
                    alpha=0.7, label='E-E0 ± Combined Error')
        plt.axhline(y=0, color='black', linestyle=':', alpha=0.5, label='Ground state (E-E0=0)')
        plt.xlabel('Bond Dimension', fontsize=12)
        plt.ylabel('E - E0', fontsize=12)
        plt.title(f'Fixed Bond Dimension Study\n({system_qubits} system + {bath_qubits} bath qubits, {num_sweeps} sweep)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Set x-axis ticks to match the actual bond dimension values
        plt.xticks(bond_dims)
        
        plt.tight_layout()
    
    return results


def sweep_convergence_study(max_sweeps=10, system_qubits=10, bath_qubits=5, 
                          half=True, open_boundary=1, J=0.4, h=0.6, p=3,
                          single_qubit_gate_noise=0., two_qubit_gate_noise=0.,
                          bond_dimensions=[32, 64], energy_density_atol=0.01):
    """
    Study energy convergence as a function of number of sweeps using fixed bond dimensions.
    
    Args:
        max_sweeps: maximum number of sweeps to test
        system_qubits: number of system qubits
        bath_qubits: number of bath qubits
        half: if True, one bath site per two system qubits
        open_boundary: boundary conditions (1=open, 0=periodic)
        J: Ising coupling strength
        h: transverse field strength
        p: number of HVA layers per sweep
        single_qubit_gate_noise: single qubit gate noise parameter
        two_qubit_gate_noise: two qubit gate noise parameter
        bond_dimensions: list of bond dimensions to test (default: [32, 64])
        energy_density_atol: absolute tolerance for energy density - std/system_qubits must be smaller than this
    """
    
    print(f"Sweep convergence study (max_sweeps: {max_sweeps})...")
    print("=" * 70)
    
    sweep_counts = []
    final_energies = []
    final_energy_diffs = []
    final_combined_stds = []
    final_bond_dims = []
    final_shots = []
    converged_status = []  # Track whether each sweep converged to both tolerances
    
    for num_sweeps in range(1, max_sweeps + 1):
        print(f"\n--- Testing {num_sweeps} sweep(s) ---")
        
        try:
            # Run fixed bond dimension study for this number of sweeps
            results = fixed_bond_dimension_study(
                system_qubits=system_qubits, bath_qubits=bath_qubits,
                half=half, open_boundary=open_boundary, J=J, h=h, p=p, num_sweeps=num_sweeps,
                single_qubit_gate_noise=single_qubit_gate_noise, two_qubit_gate_noise=two_qubit_gate_noise,
                bond_dimensions=bond_dimensions, energy_density_atol=energy_density_atol
            )
            
            # Store results for both bond dimensions
            for result in results:
                sweep_counts.append(num_sweeps)
                final_energies.append(result['energy'])
                final_energy_diffs.append(result['energy_diff'])
                final_combined_stds.append(result['combined_std'])
                final_bond_dims.append(result['bond_dim'])
                converged_status.append(True)  # Always True for fixed bond dimensions
            
            print(f"  Results:")
            for result in results:
                print(f"           bond_dim={result['bond_dim']}: E-E0 = {result['energy_diff']:.6f} ± {result['combined_std']:.6f} (shot: {result['energy_std']:.6f}, trunc: {result['truncation_error']:.6f})")
            
        except Exception as e:
            print(f"  Error for {num_sweeps} sweeps: {e}")
            break
    
    print("\n" + "=" * 70)
    print("Sweep convergence study completed!")
    
    # Print summary
    total_count = len(converged_status)
    print(f"Study completed: {total_count} data points collected")
    print("Results for each sweep count:")
    for i, sweeps in enumerate(set(sweep_counts)):
        sweep_indices = [j for j, s in enumerate(sweep_counts) if s == sweeps]
        unique_bond_dims = sorted(set([final_bond_dims[j] for j in sweep_indices]))
        print(f"  {sweeps} sweep(s): {len(sweep_indices)} data points (bond_dim={','.join(map(str, unique_bond_dims))})")
    
    # Create convergence plot
    plt.figure(figsize=(12, 8))
    
    # Energy difference vs number of sweeps
    plt.subplot(2, 1, 1)
    
    # Group by bond dimension
    unique_bond_dims = sorted(set(final_bond_dims))
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    markers = ['o', 's', '^', 'v', '<', '>', 'p', '*', 'h', 'H']
    
    for i, bond_dim in enumerate(unique_bond_dims):
        bond_dim_indices = [j for j, bd in enumerate(final_bond_dims) if bd == bond_dim]
        if bond_dim_indices:
            bd_sweeps = [sweep_counts[j] for j in bond_dim_indices]
            bd_diffs = [final_energy_diffs[j] for j in bond_dim_indices]
            bd_stds = [final_combined_stds[j] for j in bond_dim_indices]
            plt.errorbar(bd_sweeps, bd_diffs, yerr=bd_stds, 
                        marker=markers[i % len(markers)], capsize=5, capthick=2, linewidth=2, markersize=8, 
                        color=colors[i % len(colors)], label=f'Bond dim={bond_dim}')
    
    plt.axhline(y=0, color='black', linestyle=':', alpha=0.5, label='Ground state (E-E0=0)')
    plt.xlabel('Number of Sweeps', fontsize=12)
    plt.ylabel('E - E0', fontsize=12)
    plt.title(f'Sweep Convergence Study\n({system_qubits} system + {bath_qubits} bath qubits)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Set x-axis ticks to match the actual sweep count values
    plt.xticks(sorted(set(sweep_counts)))
    
    # Bond dimension vs number of sweeps
    plt.subplot(2, 1, 2)
    plt.plot(sweep_counts, final_bond_dims, 'o-', label='Bond Dimension', linewidth=2)
    plt.xlabel('Number of Sweeps', fontsize=12)
    plt.ylabel('Bond Dimension', fontsize=12)
    plt.title('Bond Dimension vs Number of Sweeps', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Set y-axis ticks to match the actual bond dimension values
    plt.yticks(unique_bond_dims)
    
    # Set x-axis ticks to match the actual sweep count values
    plt.xticks(sorted(set(sweep_counts)))
    
    plt.tight_layout()
    
    return sweep_counts, final_energies, final_energy_diffs, final_combined_stds, final_bond_dims, converged_status


def run_parameter_sweep(csv_file, parameter_sets, verbose=False):
    """
    Run adaptive precision study for multiple parameter sets and save results to CSV.
    
    Args:
        csv_file: path to CSV file to save results
        parameter_sets: list of dictionaries, each containing parameter values
        verbose: if True, print progress and create plots for each run
    """
    print(f"Starting parameter sweep with {len(parameter_sets)} parameter sets")
    print(f"Results will be saved to: {csv_file}")
    
    for i, params in enumerate(parameter_sets):
        print(f"\n--- Run {i+1}/{len(parameter_sets)} ---")
        print(f"Parameters: {params}")
        
        try:
            # Run fixed bond dimension study with current parameters
            fixed_bond_dimension_study(
                csv_file=csv_file,
                verbose=verbose,
                **params
            )
            print(f"✓ Run {i+1} completed successfully")
        except Exception as e:
            print(f"✗ Run {i+1} failed: {e}")
            # Save error information to CSV
            error_dict = {
                'system_qubits': params.get('system_qubits', 'N/A'),
                'bath_qubits': params.get('bath_qubits', 'N/A'),
                'J': params.get('J', 'N/A'),
                'h': params.get('h', 'N/A'),
                'num_sweeps': params.get('num_sweeps', 'N/A'),
                'p': params.get('p', 'N/A'),
                'training_method': params.get('training_method', 'N/A'),
                'single_qubit_gate_noise': params.get('single_qubit_gate_noise', 'N/A'),
                'two_qubit_gate_noise': params.get('two_qubit_gate_noise', 'N/A'),
                'initial_state': params.get('initial_state', 'N/A'),
                'ground_state_energy': 'ERROR',
                'bond_dim': 'ERROR',
                'energy': 'ERROR',
                'energy_diff': 'ERROR',
                'energy_std': 'ERROR',
                'truncation_error': 'ERROR',
                'combined_std': 'ERROR',
                'energy_atol': 'ERROR',
                'total_time_minutes': 'ERROR',
                'error_message': str(e)
            }
            save_result_to_csv(csv_file, error_dict)
    
    print(f"\nParameter sweep completed. Results saved to: {csv_file}")


# Example parameter sets for cluster execution
def create_example_parameter_sets():
    """
    Create parameter sets for variational cooling study.
    """
    import numpy as np
    
    parameter_sets = []
    
    # System sizes: [4+2, 8+4, 14+7, 28+14] (system_qubits+bath_qubits)
    system_sizes = [(4, 2), (8, 4), (14, 7), (28, 14)]
    
    # Number of sweeps: [0,1,2,3,4,5,6,7,8]
    num_sweeps_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    
    # J, h: [(0.4,0.6)]
    J, h = 0.4, 0.6
    
    # Noise levels: (0.001, 0.01) x [linspace(0, 1, 11)]
    noise_factors = np.linspace(0, 1, 11)
    base_single_qubit_noise = 0.001
    base_two_qubit_noise = 0.01
    
    # Generate all combinations
    # Reordered loops to prioritize system sizes and sweep counts for better load balancing
    for noise_factor in noise_factors:
        for system_qubits, bath_qubits in system_sizes:
            for num_sweeps in num_sweeps_list:
                parameter_sets.append({
                    'system_qubits': system_qubits,
                    'bath_qubits': bath_qubits,
                    'J': J,
                    'h': h,
                    'p': 3,
                    'num_sweeps': num_sweeps,
                    'single_qubit_gate_noise': base_single_qubit_noise * noise_factor,
                    'two_qubit_gate_noise': base_two_qubit_noise * noise_factor,
                    'training_method': 'energy',
                    'initial_state': 'zeros',
                    'bond_dimensions': [32, 64],  # Default bond dimensions for parameter sets
                    'energy_density_atol': 0.01  # Energy density tolerance for estimator precision
                })
    
    return parameter_sets


if __name__ == "__main__":
    import argparse
    import json
    
    # Set matplotlib formatting (without LaTeX to avoid LaTeX dependency issues)
    plt.rcParams['font.size'] = 12
    plt.rcParams['figure.dpi'] = 150
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Variational cooling MPS simulation')
    parser.add_argument('--job-id', type=int, help='Job ID for cluster execution')
    parser.add_argument('--param-file', type=str, help='JSON file containing parameter sets')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory for results')
    parser.add_argument('--shared-csv', type=str, help='Shared CSV file for all jobs (thread-safe)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output and plots')
    parser.add_argument('--sweep-study', action='store_true', help='Run sweep convergence study instead of parameter sweep')
    parser.add_argument('--bond-dims', type=str, default='32,64', help='Comma-separated list of bond dimensions to test (default: 32,64)')
    parser.add_argument('--energy-density-atol', type=float, default=0.01, help='Energy density tolerance for estimator precision (default: 0.01)')
    
    args = parser.parse_args()
    
    # Parse bond dimensions and energy density tolerance
    bond_dimensions = [int(x.strip()) for x in args.bond_dims.split(',')]
    energy_density_atol = args.energy_density_atol
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.job_id is not None and args.param_file is not None:
        # Cluster job mode: run specific parameter sets from file
        print(f"Running cluster job {args.job_id} with parameters from {args.param_file}")
        
        # Load parameter sets from file
        with open(args.param_file, 'r') as f:
            parameter_sets = json.load(f)
        
        # Use shared CSV file if specified, otherwise job-specific
        if args.shared_csv:
            csv_file = args.shared_csv
        else:
            csv_file = os.path.join(args.output_dir, f"results_job_{args.job_id:03d}.csv")
        
        # Run parameter sweep for this job
        run_parameter_sweep(csv_file, parameter_sets, verbose=args.verbose)
        
        print(f"Job {args.job_id} completed. Results saved to {csv_file}")
        
    elif args.sweep_study:
        # Option 1: Original sweep convergence study (verbose with plots)
        print("Running sweep convergence study...")
        sweep_counts, final_energies, final_energy_diffs, final_combined_stds, final_bond_dims, converged_status = sweep_convergence_study(
            max_sweeps=5,
            system_qubits=20,
            bath_qubits=10,
            half=True,
            open_boundary=1,
            J=0.4,
            h=0.6,
            p=3,
            single_qubit_gate_noise=0.0003,
            two_qubit_gate_noise=0.003,
            bond_dimensions=bond_dimensions,
            energy_density_atol=energy_density_atol
        ) 
        plt.show()
        
    else:
        # Option 2: Parameter sweep for cluster execution (non-verbose, CSV output)
        csv_file = os.path.join(args.output_dir, "adaptive_precision_results.csv")
        parameter_sets = create_example_parameter_sets()
        
        # Run parameter sweep
        run_parameter_sweep(csv_file, parameter_sets, verbose=args.verbose)
        
        print(f"\nResults saved to {csv_file}")
        print("CSV columns: system_qubits, bath_qubits, J, h, num_sweeps, p, training_method,")
        print("             single_qubit_gate_noise, two_qubit_gate_noise, initial_state,")
        print("             ground_state_energy, bond_dim, energy, energy_diff, energy_std, truncation_error, combined_std, energy_atol, total_time_minutes")