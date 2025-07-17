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


def adaptive_precision_study(energy_density_atol=0.01, system_qubits=10, bath_qubits=5, half=True, open_boundary=1, 
                           J=0.4, h=0.6, p=3, num_sweeps=1, 
                           single_qubit_gate_noise=0., two_qubit_gate_noise=0.,
                           max_timeout_minutes=30, max_bond_dim=256, min_bond_dim=8,
                           training_method="energy", initial_state="zeros", verbose=True, csv_file=None):
    """
    Adaptively adjust bond dimension to reach target precision for E-E0.
    
    Args:
        energy_density_atol: absolute tolerance for energy density - std/system_qubits must be smaller than this
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
        max_timeout_minutes: maximum runtime in minutes
        max_bond_dim: maximum bond dimension to test
        training_method: training method ("energy" or "mpo_fidelity")
        initial_state: initial state description as string (e.g., "zeros", "random", etc.)
        verbose: if True, print progress and create plots
        csv_file: path to CSV file to save results (thread-safe)
        system_qubits: number of system qubits
        bath_qubits: number of bath qubits
        half: if True, one bath site per two system qubits
        open_boundary: boundary conditions (1=open, 0=periodic)
        J: Ising coupling strength
        h: transverse field strength
        p: number of HVA layers per sweep
        num_sweeps: number of cooling sweeps
        single_qubit_gate_noise: single qubit gate noise parameter
        two_qubit_gate_noise: two qubit gate noise parameter
        max_timeout_minutes: maximum runtime in minutes
        max_bond_dim: maximum bond dimension to test
        max_shots: maximum number of shots to use
    """
    
    num_qubits = system_qubits + bath_qubits
    
    # Convert energy density tolerance to energy tolerance
    energy_atol = energy_density_atol * system_qubits
    
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
    
    if verbose:
        print(f"Tolerance criteria: energy_density_atol={energy_density_atol:.6f}, energy_atol={energy_atol:.6f} (std < energy_atol)")

    # Pre-optimized parameters for J=0.4, 3 layers
    best_para = np.array([
        0.026932368894753006, 0.58775691792609, 1.416345878671235, 
        -0.5847230425335908, 2.108254043912492, 0.9685586146293224, 
        3.141592653589793, 1.9041272021498745, -0.7655691857212363, 
        1.463995846549075, 1.062269199105236, 0.450403882505529
    ])

    # TODO: add best_para for different J, h
    
    # Split parameters
    lengths = [p, p, p, p]
    split_pts = np.cumsum(lengths)[:-1]
    
    # Create system Hamiltonian
    pauli_sys = pauli_sys_ZZ(system_qubits, 0, J, open_boundary) + pauli_sys_X(system_qubits, 0, h)
    H_sys = SparsePauliOp.from_list(pauli_sys)
    
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
    
    # Adaptive precision study
    bond_dimensions = []
    energies = []
    energy_diffs = []
    energy_stds = []
    truncation_errors = []
    combined_stds = []
    
    start_time = time.time()
    current_bond_dim = min_bond_dim # Start with minimum bond dimension
    converged = False
    dominant_variance = "truncation"
    
    while not converged:
        # Check timeout
        elapsed_time = (time.time() - start_time) / 60  # in minutes
        if elapsed_time > max_timeout_minutes:
            if verbose:
                print(f"Timeout reached ({max_timeout_minutes} minutes). Stopping.")
            break
            
        if verbose:
            print(f"Testing bond_dim={current_bond_dim}")
        
        # Setup backend with current bond dimension
        backend = get_noisy_simulator(
            single_qubit_gate_noise=single_qubit_gate_noise,
            two_qubit_gate_noise=two_qubit_gate_noise,
            max_bond_dimension=current_bond_dim,
        )
        
        # Transpile circuit
        pm = generate_preset_pass_manager(optimization_level=3, backend=backend, initial_layout=layout)
        transpiled_circuit = pm.run(total_circ)
        mapped_observables = [observable.apply_layout(transpiled_circuit.layout) for observable in observables]
        
        # Setup estimator
        # ignore this error, this line of code is correct
        options = EstimatorOptions(
            # default_shots=current_shots,
            default_precision=energy_atol/np.sqrt(len(mapped_observables))
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
        
        bond_dimensions.append(current_bond_dim)
        energies.append(energy)
        energy_diffs.append(energy_diff)
        energy_stds.append(energy_std)
        
        # Calculate truncation error as difference from previous lower bond dimension
        # Account for statistical noise in both measurements
        truncation_error = np.inf
        if len(energy_diffs) >= 2:
            # Find the most recent energy with a lower bond dimension
            for i in range(len(energy_diffs) - 2, -1, -1):  # Go backwards from second-to-last
                if bond_dimensions[i] < current_bond_dim:
                    observed_diff = abs(energy_diffs[-1] - energy_diffs[i])
                    # Isolate truncation error by subtracting statistical noise
                    # Var(difference) = Var(truncation_error) + Var(shot_noise_1) + Var(shot_noise_2)
                    current_shot_variance = energy_stds[-1]**2
                    previous_shot_variance = energy_stds[i]**2
                    if observed_diff**2 - current_shot_variance - previous_shot_variance > max(previous_shot_variance, current_shot_variance):
                        truncation_variance = observed_diff**2 - current_shot_variance - previous_shot_variance
                        dominant_variance = "truncation"
                    else:
                        truncation_variance = max(previous_shot_variance, current_shot_variance)
                        dominant_variance = "shot"
                    truncation_error = np.sqrt(truncation_variance)
                    break
            
        truncation_errors.append(truncation_error)
        
        # Combined standard deviation (shot noise + truncation error)
        combined_std = np.sqrt(energy_std**2 + truncation_error**2)
        combined_stds.append(combined_std)
        
        if verbose:
            print(f"  Energy: {energy:.6f} (E-E0: {energy_diff:.6f}) ± {energy_std:.6f}")
            print(f"  Truncation error: {truncation_error:.6f}")
            print(f"  Combined std: {combined_std:.6f}")
        
        # Check if we've reached the target
        # energy_atol: std must be smaller than energy_atol
        if combined_std <= energy_atol:
            converged = True
            if verbose:
                print(f"  ✓ Target reached! Combined std ({combined_std:.6f}) ≤ energy_atol ({energy_atol:.6f})")
        else:
            if verbose:
                print(f"  ✗ Target not reached: Combined std ({combined_std:.6f}) > energy_atol ({energy_atol:.6f})")
        if not converged:
            # Check if shot noise is dominant (we can't reduce this further)
            if dominant_variance == "shot":
                if verbose:
                    print(f"  → Shot noise is dominant. Cannot improve further. Stopping.")
                break
            else:  # Truncation error is dominant
                # Increase bond dimension
                new_bond_dim = current_bond_dim * 2
                if new_bond_dim <= max_bond_dim:
                    current_bond_dim = new_bond_dim
                    if verbose:
                        print(f"  → Increasing bond dimension to {current_bond_dim}")
                else:
                    if verbose:
                        print(f"  → Bond dimension limit would be exceeded ({new_bond_dim} > {max_bond_dim}). Stopping.")
                    break
    
    total_time = (time.time() - start_time) / 60  # in minutes
    
    if verbose:
        print("=" * 70)
        print(f"Adaptive precision study completed in {total_time:.2f} minutes!")
        print(f"Final bond dimension: {bond_dimensions[-1]}")
        print(f"Final energy: {energies[-1]:.6f} (E-E0: {energy_diffs[-1]:.6f}) ± {combined_stds[-1]:.6f}")
        print(f"Final tolerance: energy_density_atol={energy_density_atol:.6f}, energy_atol={energy_atol:.6f}")
        print(f"Achieved: std={combined_stds[-1]:.6f}")
    
    # Save results to CSV if specified
    if csv_file is not None:
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
                    'energy_density_atol': energy_density_atol,
        'max_bond_dim': max_bond_dim,
        'min_bond_dim': min_bond_dim,
        'ground_state_energy': E0,
            'final_energy': energies[-1],
            'final_bond_dim': bond_dimensions[-1],
            'final_combined_std': combined_stds[-1],
            'converged': converged,
            'total_time_minutes': total_time,
            'num_iterations': len(bond_dimensions)
        }
        save_result_to_csv(csv_file, result_dict)
    
    # Create convergence plot only if verbose
    if verbose:
        plt.figure(figsize=(12, 8))
        
        # Create subplot for energy convergence
        plt.subplot(2, 1, 1)
        plt.errorbar(range(len(energy_diffs)), energy_diffs, yerr=combined_stds, 
                    marker='o', capsize=5, capthick=2, linewidth=2, markersize=8, 
                    label='E-E0 ± Combined Error')
        plt.errorbar(range(len(energy_diffs)), energy_diffs, yerr=energy_stds, 
                    marker='s', capsize=3, capthick=1, linewidth=1, markersize=6, 
                    alpha=0.7, label='E-E0 ± Shot Noise Only')
        plt.axhline(y=0, color='black', linestyle=':', alpha=0.5, label='Ground state (E-E0=0)')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('E - E0', fontsize=12)
        plt.title(f'Adaptive Precision Study\n({system_qubits} system + {bath_qubits} bath qubits, {num_sweeps} sweep)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Create subplot for bond dimension evolution
        plt.subplot(2, 1, 2)
        plt.plot(range(len(bond_dimensions)), bond_dimensions, 'o-', label='Bond Dimension', linewidth=2)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Bond Dimension', fontsize=12)
        plt.title('Bond Dimension Evolution', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.yscale('log')
        
        # Set y-axis ticks to match the actual bond dimension values and format them as integers
        ax = plt.gca()
        ax.yaxis.set_major_locator(mticker.FixedLocator(bond_dimensions))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: str(int(x))))
        ax.yaxis.set_minor_locator(mticker.NullLocator())  # Remove minor ticks
        
        plt.tight_layout()
    
    return bond_dimensions, energies, energy_diffs, energy_stds, truncation_errors, combined_stds, converged


def sweep_convergence_study(max_sweeps=10, energy_density_atol=0.01, system_qubits=10, bath_qubits=5, 
                          half=True, open_boundary=1, J=0.4, h=0.6, p=3,
                          single_qubit_gate_noise=0., two_qubit_gate_noise=0.,
                          max_timeout_minutes=30, max_bond_dim=256):
    """
    Study energy convergence as a function of number of sweeps using adaptive precision.
    
    Args:
        max_sweeps: maximum number of sweeps to test
        atol: absolute tolerance - std must be smaller than atol
        system_qubits: number of system qubits
        bath_qubits: number of bath qubits
        half: if True, one bath site per two system qubits
        open_boundary: boundary conditions (1=open, 0=periodic)
        J: Ising coupling strength
        h: transverse field strength
        p: number of HVA layers per sweep
        single_qubit_gate_noise: single qubit gate noise parameter
        two_qubit_gate_noise: two qubit gate noise parameter
        max_timeout_minutes: maximum runtime in minutes per sweep
        max_bond_dim: maximum bond dimension to test
        max_shots: maximum number of shots to use
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
            # Run adaptive precision study for this number of sweeps
            bond_dimensions, energies, energy_diffs, energy_stds, truncation_errors, combined_stds, converged = adaptive_precision_study(
                energy_density_atol=energy_density_atol, system_qubits=system_qubits, bath_qubits=bath_qubits,
                half=half, open_boundary=open_boundary, J=J, h=h, p=p, num_sweeps=num_sweeps,
                single_qubit_gate_noise=single_qubit_gate_noise, two_qubit_gate_noise=two_qubit_gate_noise,
                max_timeout_minutes=max_timeout_minutes, max_bond_dim=max_bond_dim
            )
            
            # Store final results
            sweep_counts.append(num_sweeps)
            final_energies.append(energies[-1])
            final_energy_diffs.append(energy_diffs[-1])
            final_combined_stds.append(combined_stds[-1])
            final_bond_dims.append(bond_dimensions[-1])
            converged_status.append(converged)
            
            print(f"  Final result: E-E0 = {energy_diffs[-1]:.6f} ± {combined_stds[-1]:.6f}")
            
        except Exception as e:
            print(f"  Error for {num_sweeps} sweeps: {e}")
            break
    
    print("\n" + "=" * 70)
    print("Sweep convergence study completed!")
    
    # Print convergence summary
    converged_count = sum(converged_status)
    total_count = len(converged_status)
    print(f"Convergence summary: {converged_count}/{total_count} sweeps converged to energy density tolerance (energy_density_atol={energy_density_atol:.6f})")
    for i, (sweeps, conv) in enumerate(zip(sweep_counts, converged_status)):
        status = "✓" if conv else "✗"
        print(f"  {sweeps} sweep(s): {status}")
    
    # Create convergence plot
    plt.figure(figsize=(12, 8))
    
    # Energy difference vs number of sweeps
    plt.subplot(2, 1, 1)
    
    # Separate converged and non-converged points
    converged_indices = [i for i, conv in enumerate(converged_status) if conv]
    non_converged_indices = [i for i, conv in enumerate(converged_status) if not conv]
    
    # Plot converged points in blue
    if converged_indices:
        converged_sweeps = [sweep_counts[i] for i in converged_indices]
        converged_diffs = [final_energy_diffs[i] for i in converged_indices]
        converged_stds = [final_combined_stds[i] for i in converged_indices]
        plt.errorbar(converged_sweeps, converged_diffs, yerr=converged_stds, 
                    marker='o', capsize=5, capthick=2, linewidth=2, markersize=8, 
                    color='blue', label='Converged')
    
    # Plot non-converged points in red
    if non_converged_indices:
        non_converged_sweeps = [sweep_counts[i] for i in non_converged_indices]
        non_converged_diffs = [final_energy_diffs[i] for i in non_converged_indices]
        non_converged_stds = [final_combined_stds[i] for i in non_converged_indices]
        plt.errorbar(non_converged_sweeps, non_converged_diffs, yerr=non_converged_stds, 
                    marker='o', capsize=5, capthick=2, linewidth=2, markersize=8, 
                    color='red', label='Not Converged')
    
    plt.axhline(y=0, color='black', linestyle=':', alpha=0.5, label='Ground state (E-E0=0)')
    plt.xlabel('Number of Sweeps', fontsize=12)
    plt.ylabel('E - E0', fontsize=12)
    plt.title(f'Sweep Convergence Study\n({system_qubits} system + {bath_qubits} bath qubits)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Set x-axis ticks to match the actual sweep count values
    plt.xticks(sweep_counts)
    
    # Bond dimension vs number of sweeps
    plt.subplot(2, 1, 2)
    plt.plot(sweep_counts, final_bond_dims, 'o-', label='Final Bond Dimension', linewidth=2)
    plt.xlabel('Number of Sweeps', fontsize=12)
    plt.ylabel('Bond Dimension', fontsize=12)
    plt.title('Bond Dimension vs Number of Sweeps', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.yscale('log')
    
    # Set y-axis ticks to match the actual bond dimension values and format them as integers
    ax = plt.gca()
    ax.yaxis.set_major_locator(mticker.FixedLocator(final_bond_dims))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: str(int(x))))
    ax.yaxis.set_minor_locator(mticker.NullLocator())  # Remove minor ticks
    
    # Set x-axis ticks to match the actual sweep count values
    plt.xticks(sweep_counts)
    
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
            # Run adaptive precision study with current parameters
            adaptive_precision_study(
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
                'energy_density_atol': params.get('energy_density_atol', 'N/A'),
                'max_bond_dim': params.get('max_bond_dim', 'N/A'),
                'min_bond_dim': params.get('min_bond_dim', 'N/A'),
                'ground_state_energy': 'ERROR',
                'final_energy': 'ERROR',
                'final_bond_dim': 'ERROR',
                'final_combined_std': 'ERROR',
                'converged': 'ERROR',
                'total_time_minutes': 'ERROR',
                'num_iterations': 'ERROR',
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
                    'energy_density_atol': 0.01,
                    'system_qubits': system_qubits,
                    'bath_qubits': bath_qubits,
                    'J': J,
                    'h': h,
                    'p': 3,
                    'num_sweeps': num_sweeps,
                    'single_qubit_gate_noise': base_single_qubit_noise * noise_factor,
                    'two_qubit_gate_noise': base_two_qubit_noise * noise_factor,
                    'max_timeout_minutes': 30,
                    'max_bond_dim': 64,
                    'min_bond_dim': 8,
                    'training_method': 'energy',
                    'initial_state': 'zeros'
                })
    
    return parameter_sets


if __name__ == "__main__":
    import argparse
    import json
    
    # Set matplotlib to use LaTeX for better formatting
    plt.rcParams['text.usetex'] = True
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
    
    args = parser.parse_args()
    
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
            energy_density_atol=0.01,
            system_qubits=20,
            bath_qubits=10,
            half=True,
            open_boundary=1,
            J=0.4,
            h=0.6,
            p=3,
            single_qubit_gate_noise=0.0003,
            two_qubit_gate_noise=0.003,
            max_timeout_minutes=30,
            max_bond_dim=64
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
        print("             energy_density_atol, max_bond_dim, ground_state_energy, final_energy,")
        print("             final_bond_dim, final_combined_std, converged, total_time_minutes, num_iterations")