#!/usr/bin/env python3
"""
Data analysis module for variational cooling MPS simulation.
Creates plots and analyzes data collected by data_collection.py.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from typing import List, Tuple, Dict, Any, Optional
import argparse


def load_variational_cooling_data(filepath: str) -> Dict[str, Any]:
    """
    Load variational cooling data from JSON file.
    
    Args:
        filepath: path to the JSON file
        
    Returns:
        dict: loaded data
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def load_ground_state_data(filepath: str) -> Dict[str, Any]:
    """
    Load ground state data from JSON file.
    
    Args:
        filepath: path to the JSON file
        
    Returns:
        dict: loaded data
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def calculate_energy_density(measurements: Dict[str, Any], J: float, h: float, 
                           system_qubits: int) -> Tuple[float, float, float]:
    """
    Calculate energy density from measurements.
    
    Args:
        measurements: measurements for a specific sweep
        J: Ising coupling strength
        h: transverse field strength
        system_qubits: number of system qubits
        
    Returns:
        tuple: (energy_density, truncation_error, std_error)
    """
    # Calculate energy from ZZ correlations and X expectations
    # Note: The circuit uses RZZ(-J*alpha) and RX(-h*beta), which corresponds to
    # Hamiltonian H = -J * Σ Z_i Z_{i+1} - h * Σ X_i
    energy = 0.0
    truncation_error_squared = 0.0  # Accumulate squared errors for RSS
    std_error_squared = 0.0         # Accumulate squared errors for RSS
    
    # ZZ terms (nearest neighbor interactions) - note the negative sign
    for i in range(system_qubits - 1):
        label = f"ZZ_{i}_{i+1}"
        if label in measurements:
            obs_data = measurements[label]
            energy += -J * obs_data['mean']  # Negative sign to match circuit
            # Square the weighted error for RSS combination
            truncation_error_squared += (J * obs_data['truncation_error']) ** 2
            std_error_squared += (J * obs_data['std_error']) ** 2
                
    # Transverse field terms - note the negative sign
    for i in range(system_qubits):
        label = f"X_{i}"
        if label in measurements:
            obs_data = measurements[label]
            energy += -h * obs_data['mean']  # Negative sign to match circuit
            # Square the weighted error for RSS combination
            truncation_error_squared += (h * obs_data['truncation_error']) ** 2
            std_error_squared += (h * obs_data['std_error']) ** 2
    
    # Convert to energy density
    energy_density = energy / system_qubits
    
    # Take square root for final error (RSS method)
    truncation_error = np.sqrt(truncation_error_squared) / system_qubits
    std_error = np.sqrt(std_error_squared) / system_qubits
    
    return energy_density, truncation_error, std_error


def plot_energy_density_vs_sweeps(variational_data: Dict[str, Any], 
                                 ground_state_data: Dict[str, Any],
                                 output_dir: str = "plots") -> None:
    """
    Plot energy density above ground state as a function of sweep number.
    
    Args:
        variational_data: variational cooling data
        ground_state_data: ground state data
        output_dir: directory to save plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract metadata
    metadata = variational_data['metadata']
    final_results = variational_data['final_results']
    
    system_qubits = metadata['system_qubits']
    J = metadata['J']
    h = metadata['h']
    num_sweeps = metadata['num_sweeps']
    
    # Get ground state energy density
    ground_state_energy = ground_state_data['ground_state_results']['ground_state_energy']
    ground_state_energy_density = ground_state_energy / system_qubits
    
    # Extract measurements
    measurements = final_results['measurements']
    
    # Prepare data for plotting
    sweep_numbers = []
    energy_densities = []
    truncation_errors = []
    std_errors = []
    
    # Process each sweep
    for sweep in range(num_sweeps + 1):  # 0 to num_sweeps
        sweep_key = f"sweep_{sweep}"
        if sweep_key in measurements:
            energy_density, trunc_err, std_err = calculate_energy_density(
                measurements[sweep_key], J, h, system_qubits
            )
            
            sweep_numbers.append(sweep)
            energy_densities.append(energy_density)
            truncation_errors.append(trunc_err)
            std_errors.append(std_err)
    
    # Calculate energy density above ground state
    energy_densities_above_gs = [ed - ground_state_energy_density for ed in energy_densities]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot energy density above ground state with separate error bars
    # First plot the main line and points
    plt.plot(sweep_numbers, energy_densities_above_gs, 'o-', 
             label='Energy density above ground state', color='black', markersize=6)
    
    # Add truncation error bars in red
    plt.errorbar(sweep_numbers, energy_densities_above_gs, 
                yerr=truncation_errors, fmt='none', 
                capsize=5, capthick=2, ecolor='red', 
                elinewidth=2, label='Truncation error', alpha=0.8)
    
    # Add standard error bars in blue
    plt.errorbar(sweep_numbers, energy_densities_above_gs, 
                yerr=std_errors, fmt='none', 
                capsize=3, capthick=1, ecolor='blue', 
                elinewidth=1.5, label='Standard error', alpha=0.8)
    
    # Add horizontal line at zero (ground state)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.7, label='Ground state')
    
    # Customize the plot
    plt.xlabel('Sweep Number', fontsize=12)
    plt.ylabel('Energy Density Above Ground State', fontsize=12)
    plt.title(f'Energy Density vs Sweeps\nSystem: {system_qubits} qubits, J={J}, h={h}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add error bar legend
    plt.text(0.02, 0.98, 'Error bars:\nRed: Truncation error\nBlue: Standard error', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    filename = f"energy_density_vs_sweeps_sys{system_qubits}_J{J}_h{h}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {filepath}")
    
    plt.show()


def plot_spin_spin_correlations_vs_distance(variational_data: Dict[str, Any], 
                                          ground_state_data: Dict[str, Any],
                                          output_dir: str = "plots") -> None:
    """
    Plot spin-spin correlations vs distance for three cases:
    1. Ground state connected correlations <Z_i Z_j> - <Z_i><Z_j>
    2. Last sweep raw correlations <Z_i Z_j>
    3. Last sweep connected correlations <Z_i Z_j> - <Z_i><Z_j>
    
    Args:
        variational_data: variational cooling data
        ground_state_data: ground state data
        output_dir: directory to save plots
    """
    # Extract metadata
    metadata = variational_data['metadata']
    final_results = variational_data['final_results']
    
    system_qubits = metadata['system_qubits']
    J = metadata['J']
    h = metadata['h']
    num_sweeps = metadata['num_sweeps']
    
    # Get measurements from final sweep and ground state
    final_sweep_key = f"sweep_{num_sweeps}"
    final_measurements = final_results['measurements'][final_sweep_key]
    ground_state_obs = ground_state_data['ground_state_results']['observables']
    
    # Generate site pairs using the alternating center-outward pattern
    site_pairs = []
    center = system_qubits // 2
    i, j = center, center
    
    while (i > 0) or (j < system_qubits - 1):
        # Try to decrease i
        if i > 0:
            i -= 1
            if i < j:  # Only add if i < j to avoid duplicates
                site_pairs.append((i, j))
        
        # Try to increase j
        if j < system_qubits - 1:
            j += 1
            if i < j:  # Only add if i < j to avoid duplicates
                site_pairs.append((i, j))
    
    # Calculate distances and correlations
    distances = []
    ground_state_connected = []
    final_raw = []
    final_connected = []
    
    # Error bars
    final_raw_trunc_errors = []
    final_raw_std_errors = []
    final_connected_trunc_errors = []
    final_connected_std_errors = []
    
    for i, j in site_pairs:
        distance = j - i
        distances.append(distance)
        
        # Ground state connected correlations (assuming <Z_i> = <Z_j> = 0 for ground state)
        # This is a simplification - in general we'd need individual Z expectations
        gs_label = f"ZZ_{i}_{j}"
        if gs_label in ground_state_obs:
            gs_corr = ground_state_obs[gs_label]['value']
            # For now, assume <Z_i> = <Z_j> = 0 in ground state (typical for symmetric Hamiltonians)
            ground_state_connected.append(gs_corr)
        else:
            ground_state_connected.append(0.0)
        
        # Final sweep raw correlations
        vc_label = f"ZZ_{i}_{j}"
        if vc_label in final_measurements:
            vc_data = final_measurements[vc_label]
            final_raw.append(vc_data['mean'])
            final_raw_trunc_errors.append(vc_data['truncation_error'])
            final_raw_std_errors.append(vc_data['std_error'])
        else:
            final_raw.append(0.0)
            final_raw_trunc_errors.append(0.0)
            final_raw_std_errors.append(0.0)
        
        # Final sweep connected correlations
        # Get individual Z expectations
        z_i_label = f"Z_{i}"
        z_j_label = f"Z_{j}"
        
        z_i_expect = 0.0
        z_j_expect = 0.0
        
        if z_i_label in final_measurements:
            z_i_expect = final_measurements[z_i_label]['mean']
        if z_j_label in final_measurements:
            z_j_expect = final_measurements[z_j_label]['mean']
        
        # Calculate connected correlation
        connected_corr = final_raw[-1] - z_i_expect * z_j_expect
        final_connected.append(connected_corr)
        
        # For error bars on connected correlation, we need to propagate errors
        # This is a simplified error propagation
        if z_i_label in final_measurements and z_j_label in final_measurements:
            z_i_trunc_err = final_measurements[z_i_label]['truncation_error']
            z_j_trunc_err = final_measurements[z_j_label]['truncation_error']
            z_i_std_err = final_measurements[z_i_label]['std_error']
            z_j_std_err = final_measurements[z_j_label]['std_error']
            
            # Error propagation for connected correlation
            # ∂(ZZ - Z_i * Z_j)/∂Z_i = -Z_j, ∂(ZZ - Z_i * Z_j)/∂Z_j = -Z_i
            trunc_err = np.sqrt(
                final_raw_trunc_errors[-1]**2 + 
                (z_j_expect * z_i_trunc_err)**2 + 
                (z_i_expect * z_j_trunc_err)**2
            )
            std_err = np.sqrt(
                final_raw_std_errors[-1]**2 + 
                (z_j_expect * z_i_std_err)**2 + 
                (z_i_expect * z_j_std_err)**2
            )
        else:
            trunc_err = final_raw_trunc_errors[-1]
            std_err = final_raw_std_errors[-1]
        
        final_connected_trunc_errors.append(trunc_err)
        final_connected_std_errors.append(std_err)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot ground state connected correlations (no error bars, exact values)
    plt.plot(distances, ground_state_connected, 's-', 
             label='Ground state connected correlations', 
             color='green', markersize=8, linewidth=2)
    
    # Plot final sweep raw correlations with error bars
    plt.plot(distances, final_raw, 'o-', 
             label='Final sweep raw correlations', 
             color='black', markersize=6)
    
    # Add truncation error bars in red
    plt.errorbar(distances, final_raw, 
                yerr=final_raw_trunc_errors, fmt='none', 
                capsize=5, capthick=2, ecolor='red', 
                elinewidth=2, label='Truncation error', alpha=0.8)
    
    # Add standard error bars in blue
    plt.errorbar(distances, final_raw, 
                yerr=final_raw_std_errors, fmt='none', 
                capsize=3, capthick=1, ecolor='blue', 
                elinewidth=1.5, label='Standard error', alpha=0.8)
    
    # Plot final sweep connected correlations with error bars
    plt.plot(distances, final_connected, '^-', 
             label='Final sweep connected correlations', 
             color='purple', markersize=6)
    
    # Add truncation error bars in red
    plt.errorbar(distances, final_connected, 
                yerr=final_connected_trunc_errors, fmt='none', 
                capsize=5, capthick=2, ecolor='red', 
                elinewidth=2, alpha=0.8)
    
    # Add standard error bars in blue
    plt.errorbar(distances, final_connected, 
                yerr=final_connected_std_errors, fmt='none', 
                capsize=3, capthick=1, ecolor='blue', 
                elinewidth=1.5, alpha=0.8)
    
    # Customize the plot
    plt.xlabel('Distance |i-j|', fontsize=12)
    plt.ylabel('Spin-Spin Correlation', fontsize=12)
    plt.title(f'Spin-Spin Correlations vs Distance\nSystem: {system_qubits} qubits, J={J}, h={h}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add error bar legend
    plt.text(0.02, 0.98, 'Error bars:\nRed: Truncation error\nBlue: Standard error', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    filename = f"spin_spin_correlations_vs_distance_sys{system_qubits}_J{J}_h{h}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Spin-spin correlation plot saved to: {filepath}")
    
    plt.show()


def analyze_variational_cooling_data(system_qubits: int, bath_qubits: int, 
                                   J: float, h: float, num_sweeps: int,
                                   single_qubit_gate_noise: float, two_qubit_gate_noise: float,
                                   results_dir: str = "results", plots_dir: str = "plots") -> None:
    """
    Analyze variational cooling data and create plots.
    
    Args:
        system_qubits: number of system qubits
        bath_qubits: number of bath qubits
        J: Ising coupling strength
        h: transverse field strength
        num_sweeps: number of cooling sweeps
        single_qubit_gate_noise: single qubit gate noise parameter
        two_qubit_gate_noise: two qubit gate noise parameter
        results_dir: directory containing results
        plots_dir: directory to save plots
    """
    print(f"Analyzing variational cooling data for {system_qubits} system qubits, {bath_qubits} bath qubits")
    print(f"Parameters: J={J}, h={h}, sweeps={num_sweeps}")
    print(f"Noise: single_qubit={single_qubit_gate_noise}, two_qubit={two_qubit_gate_noise}")
    
    # Construct filenames
    vc_filename = f"variational_cooling_data_sys{system_qubits}_bath{bath_qubits}_J{J}_h{h}_sweeps{num_sweeps}_noise{single_qubit_gate_noise}_{two_qubit_gate_noise}.json"
    gs_filename = f"ground_state_data_sys{system_qubits}_J{J}_h{h}.json"
    
    vc_filepath = os.path.join(results_dir, vc_filename)
    gs_filepath = os.path.join(results_dir, gs_filename)
    
    # Check if files exist
    if not os.path.exists(vc_filepath):
        print(f"Error: Variational cooling data file not found: {vc_filepath}")
        return
    
    if not os.path.exists(gs_filepath):
        print(f"Error: Ground state data file not found: {gs_filepath}")
        return
    
    # Load data
    print("Loading variational cooling data...")
    variational_data = load_variational_cooling_data(vc_filepath)
    
    print("Loading ground state data...")
    ground_state_data = load_ground_state_data(gs_filepath)
    
    # Create plots
    print("Creating energy density plot...")
    plot_energy_density_vs_sweeps(variational_data, ground_state_data, plots_dir)
    
    print("Creating spin-spin correlation plot...")
    plot_spin_spin_correlations_vs_distance(variational_data, ground_state_data, plots_dir)
    
    print("Analysis completed!")


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description='Analyze variational cooling data')
    
    parser.add_argument('--system_qubits', type=int, default=4, help='Number of system qubits')
    parser.add_argument('--bath_qubits', type=int, default=2, help='Number of bath qubits')
    parser.add_argument('--J', type=float, default=0.6, help='Ising coupling strength')
    parser.add_argument('--h', type=float, default=0.4, help='Transverse field strength')
    parser.add_argument('--num_sweeps', type=int, default=12, help='Number of cooling sweeps')
    parser.add_argument('--single_qubit_gate_noise', type=float, default=0.0, help='Single qubit gate noise')
    parser.add_argument('--two_qubit_gate_noise', type=float, default=0.0, help='Two qubit gate noise')
    parser.add_argument('--results_dir', type=str, default='results', help='Results directory')
    parser.add_argument('--plots_dir', type=str, default='plots', help='Plots output directory')
    
    args = parser.parse_args()
    
    # Run analysis
    analyze_variational_cooling_data(
        system_qubits=args.system_qubits,
        bath_qubits=args.bath_qubits,
        J=args.J,
        h=args.h,
        num_sweeps=args.num_sweeps,
        single_qubit_gate_noise=args.single_qubit_gate_noise,
        two_qubit_gate_noise=args.two_qubit_gate_noise,
        results_dir=args.results_dir,
        plots_dir=args.plots_dir
    )


if __name__ == "__main__":
    # Check if command line arguments were provided
    import sys
    if len(sys.argv) > 1:
        # Use command line arguments
        main()
    else:
        # Example usage without command line arguments
        print("Data Analysis Module for Variational Cooling")
        print("=" * 50)
        
        # Example parameters (same as in data_collection.py)
        params = {
            'system_qubits': 10,
            'bath_qubits': 5,
            'J': 0.6,
            'h': 0.4,
            'num_sweeps': 12,
            'single_qubit_gate_noise': 0.0,
            'two_qubit_gate_noise': 0.0,
            'results_dir': 'results',
            'plots_dir': 'plots'
        }
        
        # Run analysis
        analyze_variational_cooling_data(**params)
