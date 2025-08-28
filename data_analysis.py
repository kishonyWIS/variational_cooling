#!/usr/bin/env python3
"""
Data analysis module for variational cooling MPS simulation.
Analyzes collected data and compares with ground state results.
"""

import numpy as np
import json
import os
import glob
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt

from qiskit.quantum_info import SparsePauliOp


def find_data_files(output_dir: str, system_qubits: int, bath_qubits: int, 
                   J: float, h: float, num_sweeps: int,
                   single_qubit_gate_noise: float, two_qubit_gate_noise: float) -> Tuple[str, str]:
    """
    Find the relevant variational cooling and ground state data files.
    
    Args:
        output_dir: directory containing data files
        system_qubits: number of system qubits
        bath_qubits: number of bath qubits
        J: Ising coupling strength
        h: transverse field strength
        num_sweeps: number of cooling sweeps
        single_qubit_gate_noise: single qubit gate noise parameter
        two_qubit_gate_noise: two qubit gate noise parameter
    
    Returns:
        tuple: (variational_cooling_file, ground_state_file)
    
    Raises:
        FileNotFoundError: if files cannot be found
    """
    
    # Pattern for variational cooling data
    vc_pattern = f"variational_cooling_data_sys{system_qubits}_bath{bath_qubits}_J{J}_h{h}_sweeps{num_sweeps}_noise{single_qubit_gate_noise}_{two_qubit_gate_noise}_*.json"
    
    # Pattern for ground state data
    gs_pattern = f"ground_state_data_sys{system_qubits}_J{J}_h{h}_*.json"
    
    # Find variational cooling files
    vc_files = glob.glob(os.path.join(output_dir, vc_pattern))
    if not vc_files:
        raise FileNotFoundError(f"No variational cooling data files found matching pattern: {vc_pattern}")
    
    # Find ground state files
    gs_files = glob.glob(os.path.join(output_dir, gs_pattern))
    if not gs_files:
        raise FileNotFoundError(f"No ground state data files found matching pattern: {gs_pattern}")
    
    # Get the most recent files (highest timestamp)
    vc_file = max(vc_files, key=lambda x: int(x.split('_')[-1].replace('.json', '')))
    gs_file = max(gs_files, key=lambda x: int(x.split('_')[-1].replace('.json', '')))
    
    print(f"Found variational cooling data: {os.path.basename(vc_file)}")
    print(f"Found ground state data: {os.path.basename(gs_file)}")
    
    return vc_file, gs_file


def load_data_files(vc_file: str, gs_file: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load data from JSON files.
    
    Args:
        vc_file: path to variational cooling data file
        gs_file: path to ground state data file
    
    Returns:
        tuple: (variational_cooling_data, ground_state_data)
    """
    
    with open(vc_file, 'r') as f:
        vc_data = json.load(f)
    
    with open(gs_file, 'r') as f:
        gs_data = json.load(f)
    
    return vc_data, gs_data


def create_system_hamiltonian(system_qubits: int, J: float, h: float, open_boundary: int = 1) -> SparsePauliOp:
    """
    Create the system Hamiltonian: H = -J * sum(Z_i Z_{i+1}) - h * sum(X_i)
    
    Args:
        system_qubits: number of system qubits
        J: Ising coupling strength
        h: transverse field strength
        open_boundary: boundary conditions (1=open, 0=periodic)
    
    Returns:
        SparsePauliOp: system Hamiltonian
    """
    
    pauli_terms = []
    
    # ZZ terms for nearest neighbors
    for i in range(system_qubits - open_boundary):
        pauli_str = "I" * i + "ZZ" + "I" * (system_qubits - i - 2)
        pauli_terms.append((pauli_str, -J))
    
    # X terms for all qubits
    for i in range(system_qubits):
        pauli_str = "I" * i + "X" + "I" * (system_qubits - i - 1)
        pauli_terms.append((pauli_str, -h))
    
    return SparsePauliOp.from_list(pauli_terms)


def compute_hamiltonian_expectation(observable_data: Dict[str, Any], 
                                  hamiltonian: SparsePauliOp,
                                  system_qubits: int) -> float:
    """
    Compute the expectation value of the Hamiltonian using the observable data.
    
    Args:
        observable_data: dictionary containing observable measurements
        hamiltonian: SparsePauliOp representing the system Hamiltonian
        system_qubits: number of system qubits
    
    Returns:
        float: expectation value of the Hamiltonian
    """
    
    # Extract the Hamiltonian terms
    hamiltonian_terms = hamiltonian.to_list()
    
    total_expectation = 0.0
    
    for pauli_string, coefficient in hamiltonian_terms:
        # Find the corresponding observable in the data
        if pauli_string.count('Z') == 2:
            # ZZ term - find the positions of the two Z's
            z_positions = [i for i, char in enumerate(pauli_string) if char == 'Z']
            if len(z_positions) == 2:
                i, j = z_positions
                if i < j:
                    obs_label = f"ZZ_{i}_{j}"
                else:
                    obs_label = f"ZZ_{j}_{i}"
                
                if obs_label in observable_data:
                    total_expectation += coefficient * observable_data[obs_label]['value']
                else:
                    print(f"Warning: ZZ observable {obs_label} not found in data")
        
        elif pauli_string.count('X') == 1:
            # X term - find the position of the X
            x_position = pauli_string.index('X')
            obs_label = f"X_{x_position}"
            
            if obs_label in observable_data:
                total_expectation += coefficient * observable_data[obs_label]['value']
            else:
                print(f"Warning: X observable {obs_label} not found in data")
    
    return total_expectation


def compute_observable_expectation_and_error(observable_data: Dict[str, Any], 
                                           pauli_terms: List[Tuple[str, float]],
                                           num_shots: int) -> Tuple[float, float]:
    """
    Compute expectation value and standard error for a linear combination of Pauli observables.
    
    Args:
        observable_data: dictionary containing observable measurements with 'value' and 'std' keys
        pauli_terms: list of (pauli_string, coefficient) tuples defining the observable
        num_shots: number of shots used for the measurement
    
    Returns:
        tuple: (expectation_value, standard_error)
    """
    
    total_expectation = 0.0
    total_variance = 0.0
    
    for pauli_string, coefficient in pauli_terms:
        # Find the corresponding observable in the data
        if pauli_string.count('Z') == 2:
            # ZZ term - find the positions of the two Z's
            z_positions = [i for i, char in enumerate(pauli_string) if char == 'Z']
            if len(z_positions) == 2:
                i, j = z_positions
                if i < j:
                    obs_label = f"ZZ_{i}_{j}"
                else:
                    obs_label = f"ZZ_{j}_{i}"
                
                if obs_label in observable_data:
                    value = observable_data[obs_label]['value']
                    std = observable_data[obs_label]['std']
                    total_expectation += coefficient * value
                    # Add variance contribution: (coefficient * std)^2
                    total_variance += (coefficient * std) ** 2
                else:
                    print(f"Warning: ZZ observable {obs_label} not found in data")
        
        elif pauli_string.count('X') == 1:
            # X term - find the position of the X
            x_position = pauli_string.index('X')
            obs_label = f"X_{x_position}"
            
            if obs_label in observable_data:
                value = observable_data[obs_label]['value']
                std = observable_data[obs_label]['std']
                total_expectation += coefficient * value
                # Add variance contribution: (coefficient * std)^2
                total_variance += (coefficient * std) ** 2
            else:
                print(f"Warning: X observable {obs_label} not found in data")
    
    # Standard error = sqrt(total_variance) / sqrt(num_shots)
    standard_error = np.sqrt(total_variance) / np.sqrt(num_shots)
    
    return total_expectation, standard_error


def compute_hamiltonian_expectation_with_errors(observable_data: Dict[str, Any], 
                                              hamiltonian: SparsePauliOp,
                                              system_qubits: int,
                                              num_shots: int) -> Tuple[float, float]:
    """
    Compute the expectation value of the Hamiltonian and its standard error.
    
    Args:
        observable_data: dictionary containing observable measurements
        hamiltonian: SparsePauliOp representing the system Hamiltonian
        system_qubits: number of system qubits
        num_shots: number of shots used for the measurement
    
    Returns:
        tuple: (expectation_value, standard_error)
    """
    
    # Extract the Hamiltonian terms and convert to our format
    hamiltonian_terms = hamiltonian.to_list()
    
    # Use the general function
    return compute_observable_expectation_and_error(observable_data, hamiltonian_terms, num_shots)


def analyze_energy_convergence(vc_data: Dict[str, Any], gs_data: Dict[str, Any],
                              system_qubits: int, J: float, h: float) -> Dict[str, Any]:
    """
    Analyze energy convergence by comparing variational cooling results with ground state.
    
    Args:
        vc_data: variational cooling data
        gs_data: ground state data
        system_qubits: number of system qubits
        J: Ising coupling strength
        h: transverse field strength
    
    Returns:
        dict: analysis results
    """
    
    # Create system Hamiltonian
    hamiltonian = create_system_hamiltonian(system_qubits, J, h)
    print(f"System Hamiltonian created with {len(hamiltonian.to_list())} terms")
    
    # Get ground state energy
    ground_state_energy = gs_data['ground_state_results']['ground_state_energy']
    print(f"Ground state energy: {ground_state_energy:.8f}")
    
    # Analyze each bond dimension
    analysis_results = {}
    
    for bond_dim_key, bond_dim_data in vc_data['results'].items():
        bond_dim = bond_dim_data['bond_dim']
        print(f"\nAnalyzing bond dimension {bond_dim}...")
        
        # Get initial state energy
        initial_energy = compute_hamiltonian_expectation(
            bond_dim_data['initial_state_measurements'], 
            hamiltonian, 
            system_qubits
        )
        
        # Get final sweep energy
        final_sweep_key = f"sweep_{vc_data['metadata']['num_sweeps'] - 1}"
        if final_sweep_key in bond_dim_data['sweep_measurements']:
            final_energy = compute_hamiltonian_expectation(
                bond_dim_data['sweep_measurements'][final_sweep_key], 
                hamiltonian, 
                system_qubits
            )
        else:
            print(f"Warning: Final sweep {final_sweep_key} not found")
            final_energy = None
        
        # Calculate energy differences from ground state
        initial_energy_diff = initial_energy - ground_state_energy if initial_energy is not None else None
        final_energy_diff = final_energy - ground_state_energy if final_energy is not None else None
        
        # Store results
        analysis_results[bond_dim_key] = {
            'bond_dim': bond_dim,
            'initial_energy': initial_energy,
            'final_energy': final_energy,
            'initial_energy_diff': initial_energy_diff,
            'final_energy_diff': final_energy_diff,
            'ground_state_energy': ground_state_energy,
            'energy_improvement': (initial_energy_diff - final_energy_diff) if (initial_energy_diff is not None and final_energy_diff is not None) else None
        }
        
        print(f"  Initial energy: {initial_energy:.8f} (diff from GS: {initial_energy_diff:.8f})")
        if final_energy is not None:
            print(f"  Final energy: {final_energy:.8f} (diff from GS: {final_energy_diff:.8f})")
            print(f"  Energy improvement: {analysis_results[bond_dim_key]['energy_improvement']:.8f}")
    
    return analysis_results


def plot_energy_density_vs_sweeps(vc_data: Dict[str, Any], gs_data: Dict[str, Any],
                                 system_qubits: int, J: float, h: float,
                                 output_dir: str = 'results') -> None:
    """
    Plot energy density above ground state as a function of number of sweeps.
    
    Args:
        vc_data: variational cooling data
        gs_data: ground state data
        system_qubits: number of system qubits
        J: Ising coupling strength
        h: transverse field strength
        output_dir: directory to save the plot
    """
    
    # Create system Hamiltonian
    hamiltonian = create_system_hamiltonian(system_qubits, J, h)
    print(f"Creating energy density plot with Hamiltonian of {len(hamiltonian.to_list())} terms")
    
    # Get ground state energy
    ground_state_energy = gs_data['ground_state_results']['ground_state_energy']
    
    # Get number of sweeps
    num_sweeps = vc_data['metadata']['num_sweeps']
    num_shots = vc_data['metadata']['num_shots']
    
    # Prepare data for plotting
    sweep_numbers = list(range(0, num_sweeps + 1))  # 0 for initial state, 1 to num_sweeps for sweeps
    bond_dimensions = []
    
    # Get bond dimensions from the data
    for bond_dim_key in vc_data['results'].keys():
        bond_dim = vc_data['results'][bond_dim_key]['bond_dim']
        if bond_dim not in bond_dimensions:
            bond_dimensions.append(bond_dim)
    
    bond_dimensions.sort()  # Ensure they're in order
    
    if len(bond_dimensions) < 2:
        print("Warning: Need at least 2 bond dimensions to compute truncation error")
        return
    
    # Data structures for plotting
    energy_densities = {bd: [] for bd in bond_dimensions}
    shot_errors = {bd: [] for bd in bond_dimensions}
    truncation_errors = {bd: [] for bd in bond_dimensions}
    combined_errors = {bd: [] for bd in bond_dimensions}
    
    print(f"Computing energy densities for {len(sweep_numbers)} measurement points...")
    
    # Compute energy density for each sweep and bond dimension
    for sweep_idx, sweep_num in enumerate(sweep_numbers):
        if sweep_num == 0:
            # Initial state
            sweep_key = 'initial_state_measurements'
            sweep_label = 'Initial'
        else:
            # Regular sweep
            sweep_key = f'sweep_{sweep_num - 1}'
            sweep_label = f'Sweep {sweep_num}'
        
        print(f"  Processing {sweep_label}...")
        
        # Compute energy density for each bond dimension
        for bond_dim in bond_dimensions:
            bond_dim_key = f"bond_dim_{bond_dim}"
            if bond_dim_key not in vc_data['results']:
                print(f"    Warning: Bond dimension {bond_dim} not found in data")
                continue
            
            if sweep_num == 0:
                # Initial state
                sweep_data = vc_data['results'][bond_dim_key][sweep_key]
            else:
                # Regular sweep - access through sweep_measurements
                sweep_data = vc_data['results'][bond_dim_key]['sweep_measurements'][sweep_key]
            
            # Use our general function to compute Hamiltonian expectation value and shot error
            energy, shot_error = compute_hamiltonian_expectation_with_errors(
                sweep_data, hamiltonian, system_qubits, num_shots
            )
            
            # Convert to energy density (energy per system qubit)
            energy_density = energy / system_qubits
            energy_density_above_gs = energy_density - (ground_state_energy / system_qubits)
            
            # Extract real part for plotting
            energy_density_above_gs_real = np.real(energy_density_above_gs)
            shot_error_real = np.real(shot_error / system_qubits)
            
            # Store results
            energy_densities[bond_dim].append(energy_density_above_gs_real)
            shot_errors[bond_dim].append(shot_error_real)  # Scale error by system size
        
        # Compute truncation error for this sweep
        if len(bond_dimensions) >= 2:
            # Use the difference between bond dimension 32 and 64 as truncation error
            # Take the value from the higher bond dimension (64)
            bd_low = min(bond_dimensions)
            bd_high = max(bond_dimensions)
            
            if bd_low in energy_densities and bd_high in energy_densities:
                if len(energy_densities[bd_low]) > sweep_idx and len(energy_densities[bd_high]) > sweep_idx:
                    # Truncation error is the difference between bond dimensions
                    trunc_error = abs(energy_densities[bd_low][sweep_idx] - energy_densities[bd_high][sweep_idx])
                    
                    # Store truncation error for both bond dimensions
                    for bd in bond_dimensions:
                        if len(truncation_errors[bd]) <= sweep_idx:
                            truncation_errors[bd].append(trunc_error)
                        else:
                            truncation_errors[bd][sweep_idx] = trunc_error
                    
                    # Compute combined error: sqrt(shot_error^2 + truncation_error^2)
                    for bd in bond_dimensions:
                        if len(shot_errors[bd]) > sweep_idx and len(truncation_errors[bd]) > sweep_idx:
                            shot_err = shot_errors[bd][sweep_idx]
                            trunc_err = truncation_errors[bd][sweep_idx]
                            combined_err = np.sqrt(shot_err**2 + trunc_err**2)
                            
                            if len(combined_errors[bd]) <= sweep_idx:
                                combined_errors[bd].append(combined_err)
                            else:
                                combined_errors[bd][sweep_idx] = combined_err
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot energy density vs sweeps for the higher bond dimension only (64)
    # Use the lower bond dimension (32) only for truncation error computation
    bd_low = min(bond_dimensions)  # 32
    bd_high = max(bond_dimensions)  # 64
    
    # Only plot the higher bond dimension (64)
    if bd_high in energy_densities and len(energy_densities[bd_high]) == len(sweep_numbers):
        # Plot with combined error bars (shot noise + truncation error)
        if len(combined_errors[bd_high]) == len(sweep_numbers):
            plt.errorbar(sweep_numbers, energy_densities[bd_high], 
                       yerr=combined_errors[bd_high],
                       marker='o', capsize=5, capthick=2, linewidth=2, 
                       markersize=8, color='blue', 
                       label=f'Bond dim={bd_high} (Combined error)')
        else:
            # Fallback to shot error only if combined error not available
            plt.errorbar(sweep_numbers, energy_densities[bd_high], 
                       yerr=shot_errors[bd_high],
                       marker='o', capsize=5, capthick=2, linewidth=2, 
                       markersize=8, color='blue', 
                       label=f'Bond dim={bd_high} (Shot error only)')
        
        print(f"Plotted bond dimension {bd_high} with truncation error computed from bond dimension {bd_low}")
    else:
        print(f"Warning: Higher bond dimension {bd_high} data not available for plotting")
    
    # Add horizontal line at ground state
    plt.axhline(y=0, color='black', linestyle=':', alpha=0.7, linewidth=2, 
                label='Ground state (E-E₀=0)')
    
    # Customize plot
    plt.xlabel('Sweep Number (Initial = 0)', fontsize=14)
    plt.ylabel('Energy Density Above Ground State (E-E₀)/N', fontsize=14)
    plt.title(f'Energy Density Convergence vs Number of Sweeps\n'
              f'System: {system_qubits} qubits, J={J}, h={h}', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Set x-axis ticks
    plt.xticks(sweep_numbers)
    
    # Add text box with error information
    error_info = f"Error bars include:\n"
    error_info += f"• Shot noise: std/√{num_shots}\n"
    error_info += f"• Truncation error: |E(bd={bd_low}) - E(bd={bd_high})|\n"
    error_info += f"• Plot shows bond dimension {bd_high} only"
    
    plt.text(0.02, 0.98, error_info, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    timestamp = int(time.time())
    plot_filename = f"energy_density_vs_sweeps_sys{system_qubits}_J{J}_h{h}_{timestamp}.png"
    plot_filepath = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
    print(f"Energy density plot saved to: {plot_filepath}")
    
    # Show plot
    plt.show()
    
    return {
        'sweep_numbers': sweep_numbers,
        'energy_densities': energy_densities,
        'shot_errors': shot_errors,
        'truncation_errors': truncation_errors,
        'combined_errors': combined_errors
    }


def analyze_zz_correlations(vc_data: Dict[str, Any], gs_data: Dict[str, Any],
                           system_qubits: int, output_dir: str = 'results') -> Dict[str, Any]:
    """
    Analyze ZZ correlations for system qubits at increasing distances from center.
    
    Args:
        vc_data: variational cooling data
        gs_data: ground state data
        system_qubits: number of system qubits
        output_dir: directory to save the plot
    
    Returns:
        dict containing correlation analysis results
    """
    
    print(f"Analyzing ZZ correlations for {system_qubits} system qubits...")
    
    # Get the final sweep data (highest bond dimension)
    bond_dimensions = []
    for bond_dim_key in vc_data['results'].keys():
        bond_dim = vc_data['results'][bond_dim_key]['bond_dim']
        if bond_dim not in bond_dimensions:
            bond_dimensions.append(bond_dim)
    
    bond_dimensions.sort()
    final_bond_dim = max(bond_dimensions)
    final_bond_dim_key = f"bond_dim_{final_bond_dim}"
    
    # Get final sweep measurements
    final_sweep_key = f'sweep_{len(vc_data["results"][final_bond_dim_key]["sweep_measurements"]) - 1}'
    final_sweep_data = vc_data['results'][final_bond_dim_key]['sweep_measurements'][final_sweep_key]
    
    # Generate correlation pairs at increasing distances from center
    center = system_qubits // 2
    correlation_pairs = []
    distances = []
    
    # Start with i=j=center (distance 0) - but only if we have this observable
    i, j = center, center
    # Note: self-correlations like ZZ_5_5 don't exist in the data
    # So we start with the first actual correlation pair
    
    # Then alternate between decreasing i and increasing j
    while (i > 0) or (j < system_qubits - 1):
        if i > 0:
            i -= 1
            correlation_pairs.append((i, j))
            distances.append(abs(i - j))
        
        if j < system_qubits - 1:
            j += 1
            correlation_pairs.append((i, j))
            distances.append(abs(i - j))
    
    print(f"Generated {len(correlation_pairs)} correlation pairs with distances: {distances}")
    
    # Compute correlations for variational cooling final state
    vc_raw_correlations = []
    vc_raw_correlations_errors = []
    vc_connected_correlations = []
    vc_connected_correlations_errors = []
    
    # Get single-qubit Z expectations for connected correlations
    single_z_expectations = {}
    single_z_errors = {}
    
    for i in range(system_qubits):
        obs_label = f"Z_{i}"
        if obs_label in final_sweep_data:
            single_z_expectations[i] = final_sweep_data[obs_label]['value']
            single_z_errors[i] = final_sweep_data[obs_label]['std']
        else:
            print(f"Warning: Single-qubit Z observable {obs_label} not found")
            single_z_expectations[i] = 0.0
            single_z_errors[i] = 0.0
    
    # Compute correlations for each pair
    for i, j in correlation_pairs:
        # Raw correlation <Z_i Z_j>
        obs_label = f"ZZ_{min(i,j)}_{max(i,j)}"
        if obs_label in final_sweep_data:
            raw_corr = final_sweep_data[obs_label]['value']
            raw_corr_error = final_sweep_data[obs_label]['std']
            vc_raw_correlations.append(raw_corr)
            vc_raw_correlations_errors.append(raw_corr_error)
            
            # Connected correlation <Z_i Z_j> - <Z_i><Z_j>
            if i in single_z_expectations and j in single_z_expectations:
                connected_corr = raw_corr - single_z_expectations[i] * single_z_expectations[j]
                vc_connected_correlations.append(connected_corr)
                
                # Error propagation for connected correlation
                # Var(connected) = Var(raw_corr) + Var(<Z_i><Z_j>)
                # Var(<Z_i><Z_j>) = <Z_i>^2 * Var(<Z_j>) + <Z_j>^2 * Var(<Z_i>)
                var_raw = raw_corr_error**2
                var_product = (single_z_expectations[i]**2) * (single_z_errors[j]**2) + \
                             (single_z_expectations[j]**2) * (single_z_errors[i]**2)
                connected_error = np.sqrt(var_raw + var_product)
                vc_connected_correlations_errors.append(connected_error)
            else:
                vc_connected_correlations.append(0.0)
                vc_connected_correlations_errors.append(0.0)
        else:
            print(f"Warning: ZZ observable {obs_label} not found")
            vc_raw_correlations.append(0.0)
            vc_raw_correlations_errors.append(0.0)
            vc_connected_correlations.append(0.0)
            vc_connected_correlations_errors.append(0.0)
    
    # Get ground state correlations
    gs_correlations = []
    if 'observables' in gs_data['ground_state_results']:
        gs_obs = gs_data['ground_state_results']['observables']
        # Extract ZZ correlations in the same order as our correlation pairs
        for i, j in correlation_pairs:
            obs_label = f"ZZ_{min(i,j)}_{max(i,j)}"
            if obs_label in gs_obs:
                gs_correlations.append(gs_obs[obs_label]['value'])
            else:
                print(f"Warning: Ground state observable {obs_label} not found")
                gs_correlations.append(0.0)
        print(f"Found {len(gs_correlations)} ground state ZZ correlations")
    else:
        print("Warning: No ground state observables found")
        gs_correlations = [0.0] * len(correlation_pairs)
    
    # Create the correlation plot
    plt.figure(figsize=(12, 8))
    
    # Plot raw correlations
    plt.errorbar(distances, vc_raw_correlations, yerr=vc_raw_correlations_errors,
                marker='o', capsize=5, capthick=2, linewidth=2, markersize=8,
                color='blue', label='Variational Cooling: <Z_i Z_j>')
    
    # Plot connected correlations
    plt.errorbar(distances, vc_connected_correlations, yerr=vc_connected_correlations_errors,
                marker='s', capsize=5, capthick=2, linewidth=2, markersize=8,
                color='red', label='Variational Cooling: <Z_i Z_j> - <Z_i><Z_j>')
    
    # Plot ground state correlations
    plt.plot(distances, gs_correlations, marker='^', linewidth=2, markersize=10,
             color='green', label='Ground State: <Z_i Z_j>')
    
    # Try to compute and plot ground state connected correlations if we have single-qubit Z data
    gs_connected_correlations = []
    if 'observables' in gs_data['ground_state_results']:
        gs_obs = gs_data['ground_state_results']['observables']
        # Check if we have single-qubit Z observables
        has_single_z = any(key.startswith('Z_') and not key.startswith('ZZ_') for key in gs_obs.keys())
        
        if has_single_z:
            # Extract single-qubit Z expectations
            gs_single_z = {}
            for i in range(system_qubits):
                obs_label = f"Z_{i}"
                if obs_label in gs_obs:
                    gs_single_z[i] = gs_obs[obs_label]['value']
                else:
                    gs_single_z[i] = 0.0
            
            # Compute connected correlations
            for i, j in correlation_pairs:
                obs_label = f"ZZ_{min(i,j)}_{max(i,j)}"
                if obs_label in gs_obs and i in gs_single_z and j in gs_single_z:
                    connected_corr = gs_obs[obs_label]['value'] - gs_single_z[i] * gs_single_z[j]
                    gs_connected_correlations.append(connected_corr)
                else:
                    gs_connected_correlations.append(0.0)
            
            # Plot ground state connected correlations
            plt.plot(distances, gs_connected_correlations, marker='v', linewidth=2, markersize=10,
                     color='darkgreen', linestyle='--', label='Ground State: <Z_i Z_j> - <Z_i><Z_j>')
        else:
            print("Note: Ground state data doesn't contain single-qubit Z observables for connected correlations")
    else:
        print("Note: Ground state data doesn't contain observables for connected correlations")
    
    # Customize plot
    plt.xlabel('Distance |i-j|', fontsize=14)
    plt.ylabel('Correlation Value', fontsize=14)
    plt.title(f'ZZ Correlations vs Distance\n'
              f'System: {system_qubits} qubits, Final Sweep', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Set x-axis ticks
    plt.xticks(distances)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = int(time.time())
    plot_filename = f"zz_correlations_vs_distance_sys{system_qubits}_{timestamp}.png"
    plot_filepath = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
    print(f"ZZ correlations plot saved to: {plot_filepath}")
    
    # Show plot
    plt.show()
    
    return {
        'distances': distances,
        'correlation_pairs': correlation_pairs,
        'vc_raw_correlations': vc_raw_correlations,
        'vc_raw_correlations_errors': vc_raw_correlations_errors,
        'vc_connected_correlations': vc_connected_correlations,
        'vc_connected_correlations_errors': vc_connected_correlations_errors,
        'gs_correlations': gs_correlations
    }


def main():
    """Main function to run energy convergence analysis."""
    
    # Parameters for analysis
    params = {
        'output_dir': 'results',
        'system_qubits': 10,
        'bath_qubits': 5,
        'J': 0.6,
        'h': 0.4,
        'num_sweeps': 12,
        'single_qubit_gate_noise': 0.0,
        'two_qubit_gate_noise': 0.0
    }
    
    print("=" * 60)
    print("Variational Cooling Energy Convergence Analysis")
    print("=" * 60)
    print(f"Parameters: {params}")
    
    try:
        # Find data files
        vc_file, gs_file = find_data_files(**params)
        
        # Load data
        vc_data, gs_data = load_data_files(vc_file, gs_file)
        
        # Analyze energy convergence
        analysis_results = analyze_energy_convergence(
            vc_data, gs_data, 
            params['system_qubits'], 
            params['J'], 
            params['h']
        )
        
        # Print summary
        print("\n" + "=" * 60)
        print("Energy Convergence Analysis Summary")
        print("=" * 60)
        
        for bond_dim_key, results in analysis_results.items():
            print(f"\nBond Dimension {results['bond_dim']}:")
            print(f"  Initial energy: {results['initial_energy']:.8f}")
            print(f"  Final energy: {results['final_energy']:.8f}")
            print(f"  Energy improvement: {results['energy_improvement']:.8f}")
        
        # Create energy density plot
        print("\n" + "=" * 60)
        print("Creating Energy Density vs Sweeps Plot")
        print("=" * 60)
        
        plot_data = plot_energy_density_vs_sweeps(
            vc_data, gs_data,
            params['system_qubits'],
            params['J'], params['h'],
            params['output_dir']
        )
        
        # Analyze ZZ correlations
        print("\n" + "=" * 60)
        print("Analyzing ZZ Correlations")
        print("=" * 60)
        
        correlation_data = analyze_zz_correlations(
            vc_data, gs_data,
            params['system_qubits'],
            params['output_dir']
        )
        
        # Save analysis results
        timestamp = int(time.time())
        analysis_filename = f"energy_analysis_sys{params['system_qubits']}_bath{params['bath_qubits']}_J{params['J']}_h{params['h']}_{timestamp}.json"
        analysis_filepath = os.path.join(params['output_dir'], analysis_filename)
        
        with open(analysis_filepath, 'w') as f:
            json.dump({
                'metadata': params,
                'analysis_results': analysis_results,
                'plot_data': plot_data,
                'correlation_data': correlation_data,
                'timestamp': timestamp
            }, f, indent=2, default=str)
        
        print(f"\nAnalysis results saved to: {analysis_filepath}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import time
    main()
