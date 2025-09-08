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
from scipy.optimize import curve_fit


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
                           system_qubits: int) -> Tuple[float, float]:
    """
    Calculate energy density from measurements.
    
    Args:
        measurements: measurements for a specific sweep
        J: Ising coupling strength
        h: transverse field strength
        system_qubits: number of system qubits
        
    Returns:
        tuple: (energy_density, total_error)
    """
    # Calculate energy from ZZ correlations and X expectations
    # Note: The circuit uses RZZ(-J*alpha) and RX(-h*beta), which corresponds to
    # Hamiltonian H = -J * Σ Z_i Z_{i+1} - h * Σ X_i
    energy = 0.0
    total_error_squared = 0.0  # Accumulate squared errors for RSS
    
    # ZZ terms (nearest neighbor interactions) - note the negative sign
    for i in range(system_qubits - 1):
        label = f"ZZ_{i}_{i+1}"
        if label in measurements:
            obs_data = measurements[label]
            energy += -J * obs_data['mean']  # Negative sign to match circuit
            # Use total_error for RSS combination
            total_error_squared += (J * obs_data['total_error']) ** 2
                
    # Transverse field terms - note the negative sign
    for i in range(system_qubits):
        label = f"X_{i}"
        if label in measurements:
            obs_data = measurements[label]
            energy += -h * obs_data['mean']  # Negative sign to match circuit
            # Use total_error for RSS combination
            total_error_squared += (h * obs_data['total_error']) ** 2
    
    # Convert to energy density
    energy_density = energy / system_qubits
    
    # Take square root for final error (RSS method)
    total_error = np.sqrt(total_error_squared) / system_qubits
    
    return energy_density, total_error


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
    total_errors = []
    
    # Process each sweep
    for sweep in range(num_sweeps + 1):  # 0 to num_sweeps
        sweep_key = f"sweep_{sweep}"
        if sweep_key in measurements:
            energy_density, total_err = calculate_energy_density(
                measurements[sweep_key], J, h, system_qubits
            )
            
            sweep_numbers.append(sweep)
            energy_densities.append(energy_density)
            total_errors.append(total_err)
    
    # Calculate energy density above ground state
    energy_densities_above_gs = [ed - ground_state_energy_density for ed in energy_densities]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot energy density above ground state with separate error bars
    # First plot the main line and points
    plt.plot(sweep_numbers, energy_densities_above_gs, 'o-', 
             label='Energy density above ground state', color='black', markersize=6)
    
    # Add total error bars
    plt.errorbar(sweep_numbers, energy_densities_above_gs, 
                yerr=total_errors, fmt='none', 
                capsize=5, capthick=2, ecolor='black', 
                elinewidth=2, alpha=0.8)
    
    # Add horizontal line at zero (ground state)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.7, label='Ground state')
    
    # Customize the plot
    plt.xlabel('Sweep Number', fontsize=12)
    plt.ylabel('Energy Density Above Ground State', fontsize=12)
    plt.title(f'Energy Density vs Sweeps\nSystem: {system_qubits} qubits, J={J}, h={h}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
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
    final_raw_total_errors = []
    final_connected_total_errors = []
    
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
            final_raw_total_errors.append(vc_data['total_error'])
        else:
            final_raw.append(0.0)
            final_raw_total_errors.append(0.0)
        
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
            z_i_total_err = final_measurements[z_i_label]['total_error']
            z_j_total_err = final_measurements[z_j_label]['total_error']
            
            # Error propagation for connected correlation
            # ∂(ZZ - Z_i * Z_j)/∂Z_i = -Z_j, ∂(ZZ - Z_i * Z_j)/∂Z_j = -Z_i
            total_err = np.sqrt(
                final_raw_total_errors[-1]**2 + 
                (z_j_expect * z_i_total_err)**2 + 
                (z_i_expect * z_j_total_err)**2
            )
        else:
            total_err = final_raw_total_errors[-1]
        
        final_connected_total_errors.append(total_err)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot ground state connected correlations (no error bars, exact values)
    plt.plot(distances, ground_state_connected, 's-', 
             label='Ground state: $\left\langle\sigma^Z_i \sigma^Z_j\\right\\rangle - \left\langle\sigma^Z_i\\right\\rangle \left\langle\sigma^Z_j\\right\\rangle$', 
             color='green', markersize=8, linewidth=2)
    
    # Plot final sweep raw correlations with error bars
    plt.plot(distances, final_raw, 'o-', 
             label='Steady state: $\left\langle\sigma^Z_i \sigma^Z_j\\right\\rangle$', 
             color='black', markersize=6)
    
    plt.errorbar(distances, final_raw, 
                yerr=final_raw_total_errors, fmt='none', 
                capsize=5, capthick=2, ecolor='black', 
                elinewidth=2, alpha=0.8)
    
    # Plot final sweep connected correlations with error bars
    plt.plot(distances, final_connected, '^-', 
             label='Steady state: $\left\langle\sigma^Z_i \sigma^Z_j\\right\\rangle - \left\langle\sigma^Z_i\\right\\rangle \left\langle\sigma^Z_j\\right\\rangle$', 
             color='purple', markersize=6)
    
    plt.errorbar(distances, final_connected, 
                yerr=final_connected_total_errors, fmt='none', 
                capsize=5, capthick=2, ecolor='purple', 
                elinewidth=2, alpha=0.8)

    # set x-ticks to integers
    plt.xticks(distances)
    
    # Customize the plot
    plt.xlabel('Distance |i-j|', fontsize=12)
    plt.ylabel('Spin-Spin Correlation', fontsize=12)
    plt.title(f'Spin-Spin Correlations vs Distance\nSystem: {system_qubits} qubits, J={J}, h={h}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    # Save the plot
    filename = f"spin_spin_correlations_vs_distance_sys{system_qubits}_J{J}_h{h}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Spin-spin correlation plot saved to: {filepath}")
    
    plt.show()


def plot_energy_density_vs_noise_for_different_system_sizes(results_dir: str = "results", 
                                                          output_dir: str = "plots") -> None:
    """
    Plot energy density after last sweep vs noise for different system sizes.
    Creates two panels: J=0.6,h=0.4 and J=0.4,h=0.6.
    
    Args:
        results_dir: directory containing results
        output_dir: directory to save plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the two J,h combinations to plot
    J_h_combinations = [(0.6, 0.4), (0.4, 0.6)]
    
    # Define system sizes (same as in cluster_job_generator_data_collection.py)
    system_sizes = [(4, 2), (8, 4), (12, 6), (16, 8), (20, 10), (24, 12), (28, 14)]
    
    # Define noise levels (same as in cluster_job_generator_data_collection.py)
    noise_factors = np.linspace(0, 1, 11)
    base_single_qubit_noise = 0.001
    base_two_qubit_noise = 0.01
    
    # Fixed parameters
    num_sweeps = 12
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Track all legend handles and labels for a single legend
    all_legend_handles = []
    all_legend_labels = []
    
    for panel_idx, (J, h) in enumerate(J_h_combinations):
        ax = axes[panel_idx]
        print(f"Processing panel {panel_idx + 1}/2: J={J}, h={h}")
        
        # Collect data for this J,h combination
        for sys_idx, (system_qubits, bath_qubits) in enumerate(system_sizes):
            print(f"  Processing system size: {system_qubits}+{bath_qubits} qubits")
            noise_levels = []
            energy_densities = []
            total_errors = []
            
            # Collect data for each noise level
            for noise_factor in noise_factors:
                single_qubit_noise = base_single_qubit_noise * noise_factor
                two_qubit_noise = base_two_qubit_noise * noise_factor
                
                # Construct filename
                filename = f"variational_cooling_data_sys{system_qubits}_bath{bath_qubits}_J{J}_h{h}_sweeps{num_sweeps}_noise{single_qubit_noise}_{two_qubit_noise}.json"
                filepath = os.path.join(results_dir, filename)
                
                if os.path.exists(filepath):
                    try:
                        # Load data
                        with open(filepath, 'r') as f:
                            data = json.load(f)
                        
                        # Get measurements from final sweep
                        final_sweep_key = f"sweep_{num_sweeps}"
                        measurements = data['final_results']['measurements'][final_sweep_key]
                        
                        # Calculate energy density
                        energy_density, total_err = calculate_energy_density(
                            measurements, J, h, system_qubits
                        )
                        
                        # Get ground state energy for this system size
                        gs_filename = f"ground_state_data_sys{system_qubits}_J{J}_h{h}.json"
                        gs_filepath = os.path.join(results_dir, gs_filename)
                        
                        if os.path.exists(gs_filepath):
                            try:
                                with open(gs_filepath, 'r') as f:
                                    gs_data = json.load(f)
                                ground_state_energy_density = gs_data['ground_state_results']['ground_state_energy'] / system_qubits
                                
                                # Calculate energy density above ground state
                                energy_density_above_gs = energy_density - ground_state_energy_density
                                
                                noise_levels.append(noise_factor)
                                energy_densities.append(energy_density_above_gs)
                                total_errors.append(total_err)
                            except (json.JSONDecodeError, KeyError) as e:
                                print(f"Warning: Could not process ground state file {gs_filename}: {e}")
                                continue
                        else:
                            print(f"Warning: Ground state file not found: {gs_filename}")
                            continue
                        
                    except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
                        print(f"Warning: Could not process {filename}: {e}")
                        continue
            
            if noise_levels:  # Only plot if we have data
                print(f"    Found {len(noise_levels)} data points")
                
                # Sort by noise level
                sorted_indices = np.argsort(noise_levels)
                noise_levels = [noise_levels[i] for i in sorted_indices]
                energy_densities = [energy_densities[i] for i in sorted_indices]
                total_errors = [total_errors[i] for i in sorted_indices]
                
                # Plot with error bars
                color = plt.cm.viridis(sys_idx / (len(system_sizes) - 1))
                
                # Plot main line and points
                line, = ax.plot(noise_levels, energy_densities, 'o-', 
                               color=color, markersize=6, linewidth=2)
                
                # Add total error bars
                ax.errorbar(noise_levels, energy_densities, 
                           yerr=total_errors, fmt='none', 
                           capsize=5, capthick=2, ecolor=color, 
                           elinewidth=2, alpha=0.8)
                
                # Collect legend handle and label for the first panel only
                if panel_idx == 0:
                    all_legend_handles.append(line)
                    all_legend_labels.append(f'{system_qubits}+{bath_qubits} qubits')
            else:
                print(f"    No data found for this system size")
        
        # Customize subplot
        ax.set_xlabel('Noise Factor', fontsize=12)
        ax.set_ylabel('Energy Density Above Ground State', fontsize=12)
        ax.set_title(f'J={J}, h={h}', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Set x-ticks to show noise factors clearly
        ax.set_xticks(noise_factors[::2])  # Show every other tick to avoid crowding
    
    # Add overall title
    fig.suptitle('Energy Density vs Noise for Different System Sizes', fontsize=16)
    
    # Add single legend to the first panel (top left corner)
    if all_legend_handles:
        axes[0].legend(all_legend_handles, all_legend_labels, 
                       loc='upper left', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    
    # Save the plot
    filename = "energy_density_vs_noise_for_different_system_sizes.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Energy density vs noise plot saved to: {filepath}")
    
    # Print summary
    print(f"\nPlot creation completed!")
    print(f"Two panels created: J=0.6,h=0.4 and J=0.4,h=0.6")
    print(f"System sizes included: {len(system_sizes)}")
    print(f"Noise levels: {len(noise_factors)} (0.0 to 1.0)")
    
    plt.show()


def plot_correlation_length_vs_noise_largest_system(results_dir: str = "results", 
                                                   output_dir: str = "plots", num_points_fit: int = 6) -> None:
    """
    Plot connected correlations vs distance for different noise levels for the largest system size.
    Fits exponential decay to estimate correlation length and plots correlation length vs noise.
    
    Args:
        results_dir: directory containing results
        output_dir: directory to save plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Fixed parameters for this analysis
    J = 0.6
    h = 0.4
    system_qubits = 28
    bath_qubits = 14
    num_sweeps = 12
    
    # Define noise levels
    noise_factors = np.linspace(0, 1, 11)
    base_single_qubit_noise = 0.001
    base_two_qubit_noise = 0.01
    
    # Function to fit linear decay in log space
    def linear_decay_log_space(distance, correlation_length, log_amplitude):
        """Linear decay in log space: log_amplitude - distance/correlation_length"""
        return log_amplitude - distance / correlation_length
    
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
    
    # Calculate distances
    distances = [j - i for i, j in site_pairs]
    
    # Create figure with main plot and inset
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Store correlation lengths and noise levels for inset
    correlation_lengths = []
    noise_levels_for_fit = []
    
    # Colors for different noise levels
    colors = plt.cm.viridis(np.linspace(0, 1, len(noise_factors)))
    
    print(f"Analyzing correlation length vs noise for {system_qubits}+{bath_qubits} qubits, J={J}, h={h}")
    
    for noise_idx, noise_factor in enumerate(noise_factors):
        single_qubit_noise = base_single_qubit_noise * noise_factor
        two_qubit_noise = base_two_qubit_noise * noise_factor
        
        # Construct filename
        filename = f"variational_cooling_data_sys{system_qubits}_bath{bath_qubits}_J{J}_h{h}_sweeps{num_sweeps}_noise{single_qubit_noise}_{two_qubit_noise}.json"
        filepath = os.path.join(results_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"Warning: File not found for noise factor {noise_factor}: {filename}")
            continue
        
        try:
            # Load data
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Get measurements from final sweep
            final_sweep_key = f"sweep_{num_sweeps}"
            measurements = data['final_results']['measurements'][final_sweep_key]
            
            # Calculate connected correlations
            connected_correlations = []
            valid_distances = []
            connected_total_errors = []
            
            for i, j in site_pairs:
                # Get ZZ correlation
                zz_label = f"ZZ_{i}_{j}"
                if zz_label not in measurements:
                    continue
                
                zz_data = measurements[zz_label]
                zz_corr = zz_data['mean']
                zz_total_err = zz_data['total_error']
                
                # Get individual Z expectations
                z_i_label = f"Z_{i}"
                z_j_label = f"Z_{j}"
                
                z_i_expect = 0.0
                z_j_expect = 0.0
                z_i_total_err = 0.0
                z_j_total_err = 0.0
                
                if z_i_label in measurements:
                    z_i_expect = measurements[z_i_label]['mean']
                    z_i_total_err = measurements[z_i_label]['total_error']
                if z_j_label in measurements:
                    z_j_expect = measurements[z_j_label]['mean']
                    z_j_total_err = measurements[z_j_label]['total_error']
                
                # Calculate connected correlation
                connected_corr = zz_corr - z_i_expect * z_j_expect
                connected_correlations.append(connected_corr)
                valid_distances.append(j - i)
                
                # Error propagation for connected correlation
                # ∂(ZZ - Z_i * Z_j)/∂Z_i = -Z_j, ∂(ZZ - Z_i * Z_j)/∂Z_j = -Z_i
                total_err = np.sqrt(
                    zz_total_err**2 + 
                    (z_j_expect * z_i_total_err)**2 + 
                    (z_i_expect * z_j_total_err)**2
                )
                connected_total_errors.append(total_err)
            
            if len(connected_correlations) < 3:
                print(f"Warning: Insufficient data points for noise factor {noise_factor}")
                continue
            
            # Fit linear decay in log space
            try:
                # Filter out negative correlations and corresponding distances and errors
                positive_mask = np.array(connected_correlations) > 0
                positive_correlations = np.array(connected_correlations)[positive_mask]
                positive_distances = np.array(valid_distances)[positive_mask]
                positive_total_errors = np.array(connected_total_errors)[positive_mask]
                
                if len(positive_correlations) < 3:
                    print(f"  Noise {noise_factor:.1f}: Insufficient positive correlations for fitting")
                    continue
                
                # Take logarithm of positive correlations
                log_correlations = np.log(positive_correlations)
                
                # Initial guess: correlation length ~ system size / 4, log_amplitude ~ log(max correlation)
                initial_guess = [system_qubits / 4, np.max(log_correlations)]
                
                # Fit with bounds to ensure physical values
                # Use up to num_points_fit positive correlations, or all if fewer
                fit_limit = min(num_points_fit, len(positive_correlations))
                popt, pcov = curve_fit(linear_decay_log_space, positive_distances[:fit_limit], log_correlations[:fit_limit], 
                                     p0=initial_guess, 
                                     bounds=([1, -np.inf], [system_qubits, np.inf]))
                
                correlation_length, log_amplitude = popt
                
                # Only keep reasonable fits (correlation length should be positive and not too large)
                if 0.1 < correlation_length < system_qubits:
                    correlation_lengths.append(correlation_length)
                    noise_levels_for_fit.append(noise_factor)
                    
                    # Plot the data and fit
                    color = colors[noise_idx]
                    
                    # Plot only positive correlation data points
                    ax.plot(positive_distances, positive_correlations, 'o', 
                           color=color, markersize=6, alpha=0.7)
                    
                    # Plot fitted curve in original space (only for positive correlations)
                    if len(positive_distances) > 0:
                        fit_distances = np.linspace(min(positive_distances), max(positive_distances), 100)
                        fit_log_correlations = linear_decay_log_space(fit_distances, correlation_length, log_amplitude)
                        fit_correlations = np.exp(fit_log_correlations)
                        ax.plot(fit_distances, fit_correlations, ':', 
                               color=color, linewidth=2, alpha=0.8)
                    
                    print(f"  Noise {noise_factor:.1f}: ξ = {correlation_length:.2f} ± {np.sqrt(pcov[0,0]):.2f}")
                else:
                    print(f"  Noise {noise_factor:.1f}: Fit parameters out of reasonable range")
                    
            except (RuntimeError, ValueError) as e:
                print(f"  Noise {noise_factor:.1f}: Fit failed - {e}")
                continue
                
        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            print(f"Warning: Could not process {filename}: {e}")
            continue
    
    # Customize main plot
    ax.semilogy()
    ax.set_xlabel('Distance |i-j|', fontsize=12)
    ax.set_ylabel('$\langle \sigma^Z_i \sigma^Z_j \\rangle - \langle \sigma^Z_i \\rangle \langle \sigma^Z_j \\rangle$', fontsize=12)
    ax.set_title(f'Spin-Spin Correlations vs Distance\nSystem: {system_qubits}+{bath_qubits} qubits, J={J}, h={h}', fontsize=14)
    ax.grid(True, alpha=0.3)
    # Add colorbar for noise factor
    norm = plt.Normalize(0, 1)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8, aspect=30)
    cbar.set_label('Noise Factor', fontsize=12)
    cbar.set_ticks(np.linspace(0, 1, 6))  # Show 6 ticks from 0 to 1
    
    # Create inset for correlation length vs noise
    if len(correlation_lengths) > 1:
        # bottom right
        axins = ax.inset_axes([0.1, 0.1, 0.35, 0.35])
        
        # Sort by noise level
        sorted_indices = np.argsort(noise_levels_for_fit)
        sorted_noise = [noise_levels_for_fit[i] for i in sorted_indices]
        sorted_corr_lengths = [correlation_lengths[i] for i in sorted_indices]
        
        axins.plot(sorted_noise, sorted_corr_lengths, 'o-', color='red', linewidth=2, markersize=6)
        axins.set_xlabel('Noise Factor', fontsize=10)
        axins.set_ylabel('Correlation Length ξ', fontsize=10)
        axins.set_title('Correlation Length vs Noise', fontsize=11)
        axins.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    filename = f"correlation_length_vs_noise_sys{system_qubits}_J{J}_h{h}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Correlation length vs noise plot saved to: {filepath}")
    
    # Print summary
    print(f"\nCorrelation length analysis completed!")
    print(f"System size: {system_qubits}+{bath_qubits} qubits")
    print(f"Parameters: J={J}, h={h}")
    print(f"Successful fits: {len(correlation_lengths)}/{len(noise_factors)} noise levels")
    
    plt.show()


def plot_spin_spin_correlations_largest_system_zero_noise(results_dir: str = "results", 
                                                         output_dir: str = "plots") -> None:
    """
    Plot spin-spin correlations vs distance for the largest available system size with zero noise.
    Compares variational cooling results (solid lines) to ground state values (dashed lines)
    for different J,h combinations in different colors.
    
    Args:
        results_dir: directory containing results
        output_dir: directory to save plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define J,h combinations to plot
    J_h_combinations = [(0.6, 0.4), (0.55, 0.45), (0.45, 0.55), (0.4, 0.6)]
    
    # Define system sizes to check (largest first)
    system_sizes = [(28, 14), (24, 12), (20, 10), (16, 8), (12, 6), (8, 4), (4, 2)]
    
    # Fixed parameters
    num_sweeps = 12
    single_qubit_noise = 0.0
    two_qubit_noise = 0.0
    
    # Find the largest available system size with zero noise data
    largest_available_system = None
    largest_available_bath = None
    
    for system_qubits, bath_qubits in system_sizes:
        # Check if we have data for ALL J,h combinations
        all_combinations_available = True
        for J, h in J_h_combinations:
            vc_filename = f"variational_cooling_data_sys{system_qubits}_bath{bath_qubits}_J{J}_h{h}_sweeps{num_sweeps}_noise{single_qubit_noise}_{two_qubit_noise}.json"
            gs_filename = f"ground_state_data_sys{system_qubits}_J{J}_h{h}.json"
            vc_filepath = os.path.join(results_dir, vc_filename)
            gs_filepath = os.path.join(results_dir, gs_filename)
            
            if not os.path.exists(vc_filepath) or not os.path.exists(gs_filepath):
                all_combinations_available = False
                break
        
        if all_combinations_available:
            largest_available_system = system_qubits
            largest_available_bath = bath_qubits
            break
    
    if largest_available_system is None:
        print("No zero noise data found for any system size!")
        return
    
    print(f"Using largest available system size: {largest_available_system}+{largest_available_bath} qubits")
    
    # Generate site pairs using the alternating center-outward pattern
    site_pairs = []
    center = largest_available_system // 2
    i, j = center, center
    
    while (i > 0) or (j < largest_available_system - 1):
        # Try to decrease i
        if i > 0:
            i -= 1
            if i < j:  # Only add if i < j to avoid duplicates
                site_pairs.append((i, j))
        
        # Try to increase j
        if j < largest_available_system - 1:
            j += 1
            if i < j:  # Only add if i < j to avoid duplicates
                site_pairs.append((i, j))
    
    # Calculate distances
    distances = [j - i for i, j in site_pairs]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Create colorbar based on J values (blues to reds)
    J_values = [combo[0] for combo in J_h_combinations]
    min_J, max_J = min(J_values), max(J_values)
    
    for combo_idx, (J, h) in enumerate(J_h_combinations):
        # Color based on J value (blues to reds)
        color = plt.cm.rainbow((J - min_J) / (max_J - min_J))
        
        # Check if variational cooling data exists
        vc_filename = f"variational_cooling_data_sys{largest_available_system}_bath{largest_available_bath}_J{J}_h{h}_sweeps{num_sweeps}_noise{single_qubit_noise}_{two_qubit_noise}.json"
        vc_filepath = os.path.join(results_dir, vc_filename)
        
        # Check if ground state data exists
        gs_filename = f"ground_state_data_sys{largest_available_system}_J{J}_h{h}.json"
        gs_filepath = os.path.join(results_dir, gs_filename)
        
        if not os.path.exists(vc_filepath):
            print(f"Warning: Variational cooling data not found for J={J}, h={h}")
            continue
        
        if not os.path.exists(gs_filepath):
            print(f"Warning: Ground state data not found for J={J}, h={h}")
            continue
        
        try:
            # Load variational cooling data
            with open(vc_filepath, 'r') as f:
                vc_data = json.load(f)
            
            # Load ground state data
            with open(gs_filepath, 'r') as f:
                gs_data = json.load(f)
            
            # Get measurements from final sweep
            final_sweep_key = f"sweep_{num_sweeps}"
            vc_measurements = vc_data['final_results']['measurements'][final_sweep_key]
            gs_obs = gs_data['ground_state_results']['observables']
            
            # Calculate correlations for both variational cooling and ground state
            vc_correlations = []
            gs_correlations = []
            valid_distances = []
            
            for i, j in site_pairs:
                # Get ZZ correlation
                zz_label = f"ZZ_{i}_{j}"
                if zz_label not in vc_measurements:
                    continue
                
                # Variational cooling connected correlation
                vc_zz_data = vc_measurements[zz_label]
                vc_zz_corr = vc_zz_data['mean']
                
                # Get individual Z expectations for variational cooling
                z_i_label = f"Z_{i}"
                z_j_label = f"Z_{j}"
                
                vc_z_i_expect = 0.0
                vc_z_j_expect = 0.0
                
                if z_i_label in vc_measurements:
                    vc_z_i_expect = vc_measurements[z_i_label]['mean']
                if z_j_label in vc_measurements:
                    vc_z_j_expect = vc_measurements[z_j_label]['mean']
                
                vc_connected_corr = vc_zz_corr - vc_z_i_expect * vc_z_j_expect
                
                # Ground state connected correlation
                if zz_label in gs_obs:
                    gs_zz_corr = gs_obs[zz_label]['value']
                    # For ground state, assume <Z_i> = <Z_j> = 0 (typical for symmetric Hamiltonians)
                    gs_connected_corr = gs_zz_corr
                else:
                    gs_connected_corr = 0.0
                
                vc_correlations.append(vc_connected_corr)
                gs_correlations.append(gs_connected_corr)
                valid_distances.append(j - i)
            
            if len(vc_correlations) > 0:
                # Plot variational cooling results (solid line with markers)
                ax.plot(valid_distances, vc_correlations, '-o', 
                       color=color, linewidth=2, markersize=6, 
                       label=f'J={J}, h={h}')
                
                # Plot ground state results (dashed line with markers)
                ax.plot(valid_distances, gs_correlations, '--s', 
                       color=color, linewidth=2, markersize=5, 
                       alpha=0.7)
                
                print(f"  J={J}, h={h}: {len(vc_correlations)} data points")
            else:
                print(f"  J={J}, h={h}: No valid correlations found")
                
        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            print(f"Warning: Could not process data for J={J}, h={h}: {e}")
            continue
    
    # Customize plot
    ax.set_xlabel('Distance |i-j|', fontsize=12)
    ax.set_ylabel('$\langle \sigma^Z_i \sigma^Z_j \\rangle - \langle \sigma^Z_i \\rangle \langle \sigma^Z_j \\rangle$', fontsize=12)
    ax.set_title(f'Spin-Spin Correlations vs Distance\nSystem: {largest_available_system}+{largest_available_bath} qubits, Zero Noise', fontsize=14)
    ax.grid(True, alpha=0.3)
    # Create invisible handles for legend only (no lines on plot)
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color='k', linestyle='-', label='Steady State'),
        Line2D([0], [0], color='k', linestyle='--', label='Ground State')
    ]
    ax.legend(handles=legend_handles + ax.get_legend_handles_labels()[0],
              labels=['Steady State', 'Ground State'] + ax.get_legend_handles_labels()[1],
              fontsize=10)
    
    # Set x-ticks to integers
    ax.set_xticks(distances)
    
    plt.tight_layout()
    
    # Save the plot
    filename = f"spin_spin_correlations_largest_system_zero_noise_sys{largest_available_system}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Spin-spin correlations plot saved to: {filepath}")
    
    # Print summary
    print(f"\nSpin-spin correlations analysis completed!")
    print(f"System size: {largest_available_system}+{largest_available_bath} qubits")
    print(f"J,h combinations: {len(J_h_combinations)}")
    print(f"Zero noise analysis")
    
    plt.show()


def plot_magnetization_vs_position_different_jh_zero_noise(results_dir: str = "results", 
                                                          output_dir: str = "plots") -> None:
    """
    Plot magnetization <sigma^Z_i> vs position i for different J,h combinations at zero noise.
    Uses the largest available system size.
    
    Args:
        results_dir: directory containing results
        output_dir: directory to save plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define J,h combinations to plot
    J_h_combinations = [(0.6, 0.4), (0.55, 0.45), (0.45, 0.55), (0.4, 0.6)]
    
    # Define system sizes to check (largest first)
    system_sizes = [(28, 14), (24, 12), (20, 10), (16, 8), (12, 6), (8, 4), (4, 2)]
    
    # Fixed parameters
    num_sweeps = 12
    single_qubit_noise = 0.0
    two_qubit_noise = 0.0
    
    # Find the largest available system size with zero noise data
    largest_available_system = None
    largest_available_bath = None
    
    for system_qubits, bath_qubits in system_sizes:
        # Check if we have data for ALL J,h combinations
        all_combinations_available = True
        for J, h in J_h_combinations:
            vc_filename = f"variational_cooling_data_sys{system_qubits}_bath{bath_qubits}_J{J}_h{h}_sweeps{num_sweeps}_noise{single_qubit_noise}_{two_qubit_noise}.json"
            vc_filepath = os.path.join(results_dir, vc_filename)
            
            if not os.path.exists(vc_filepath):
                all_combinations_available = False
                break
        
        if all_combinations_available:
            largest_available_system = system_qubits
            largest_available_bath = bath_qubits
            break
    
    if largest_available_system is None:
        print("No zero noise data found for any system size!")
        return
    
    print(f"Using largest available system size: {largest_available_system}+{largest_available_bath} qubits")
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Create colorbar based on J values (blues to reds)
    J_values = [combo[0] for combo in J_h_combinations]
    min_J, max_J = min(J_values), max(J_values)
    
    for combo_idx, (J, h) in enumerate(J_h_combinations):
        # Color based on J value (blues to reds)
        color = plt.cm.rainbow((J - min_J) / (max_J - min_J))
        
        # Check if variational cooling data exists
        vc_filename = f"variational_cooling_data_sys{largest_available_system}_bath{largest_available_bath}_J{J}_h{h}_sweeps{num_sweeps}_noise{single_qubit_noise}_{two_qubit_noise}.json"
        vc_filepath = os.path.join(results_dir, vc_filename)
        
        if not os.path.exists(vc_filepath):
            print(f"Warning: Variational cooling data not found for J={J}, h={h}")
            continue
        
        try:
            # Load variational cooling data
            with open(vc_filepath, 'r') as f:
                vc_data = json.load(f)
            
            # Get measurements from final sweep
            final_sweep_key = f"sweep_{num_sweeps}"
            vc_measurements = vc_data['final_results']['measurements'][final_sweep_key]
            
            # Extract magnetization data
            positions = []
            magnetizations = []
            total_errors = []
            
            for i in range(largest_available_system):
                z_label = f"Z_{i}"
                if z_label in vc_measurements:
                    z_data = vc_measurements[z_label]
                    positions.append(i)
                    magnetizations.append(z_data['mean'])
                    total_errors.append(z_data['total_error'])
            
            if len(magnetizations) > 0:
                # Plot magnetization with error bars
                ax.plot(positions, magnetizations, 'o-', 
                       color=color, linewidth=2, markersize=6, 
                       label=f'J={J}, h={h}')
                
                # Add error bars
                ax.errorbar(positions, magnetizations, 
                           yerr=total_errors, fmt='none', 
                           capsize=3, capthick=1, ecolor=color, 
                           elinewidth=1, alpha=0.6)
                
                print(f"  J={J}, h={h}: {len(magnetizations)} data points")
            else:
                print(f"  J={J}, h={h}: No magnetization data found")
                
        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            print(f"Warning: Could not process data for J={J}, h={h}: {e}")
            continue
    
    # Customize plot
    ax.set_xlabel('Position i', fontsize=12)
    ax.set_ylabel('$\langle \sigma^Z_i \\rangle$', fontsize=12)
    ax.set_title(f'Magnetization vs Position\nSystem: {largest_available_system}+{largest_available_bath} qubits, Zero Noise', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Set x-ticks to integers
    if largest_available_system:
        ax.set_xticks(range(largest_available_system))
    
    plt.tight_layout()
    
    # Save the plot
    filename = f"magnetization_vs_position_different_jh_zero_noise_sys{largest_available_system}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Magnetization vs position plot saved to: {filepath}")
    
    # Print summary
    print(f"\nMagnetization vs position analysis completed!")
    print(f"System size: {largest_available_system}+{largest_available_bath} qubits")
    print(f"J,h combinations: {len(J_h_combinations)}")
    print(f"Zero noise analysis")
    
    plt.show()


def plot_magnetization_vs_position_different_noise_fixed_jh(results_dir: str = "results", 
                                                           output_dir: str = "plots") -> None:
    """
    Plot magnetization <sigma^Z_i> vs position i for different noise levels at fixed J=0.6, h=0.4.
    Uses the largest available system size.
    
    Args:
        results_dir: directory containing results
        output_dir: directory to save plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Fixed parameters
    J = 0.6
    h = 0.4
    num_sweeps = 12
    
    # Define system sizes to check (largest first)
    system_sizes = [(28, 14), (24, 12), (20, 10), (16, 8), (12, 6), (8, 4), (4, 2)]
    
    # Define noise levels
    noise_factors = np.linspace(0, 1, 11)
    base_single_qubit_noise = 0.001
    base_two_qubit_noise = 0.01
    
    # Find the largest available system size
    largest_available_system = None
    largest_available_bath = None
    
    for system_qubits, bath_qubits in system_sizes:
        # Check if we have data for at least one noise level
        has_data = False
        for noise_factor in noise_factors:
            single_qubit_noise = base_single_qubit_noise * noise_factor
            two_qubit_noise = base_two_qubit_noise * noise_factor
            vc_filename = f"variational_cooling_data_sys{system_qubits}_bath{bath_qubits}_J{J}_h{h}_sweeps{num_sweeps}_noise{single_qubit_noise}_{two_qubit_noise}.json"
            vc_filepath = os.path.join(results_dir, vc_filename)
            
            if os.path.exists(vc_filepath):
                has_data = True
                break
        
        if has_data:
            largest_available_system = system_qubits
            largest_available_bath = bath_qubits
            break
    
    if largest_available_system is None:
        print("No data found for any system size!")
        return
    
    print(f"Using largest available system size: {largest_available_system}+{largest_available_bath} qubits")
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Colors for different noise levels
    colors = plt.cm.viridis(np.linspace(0, 1, len(noise_factors)))
    
    print(f"Analyzing magnetization vs position for different noise levels")
    print(f"System: {largest_available_system}+{largest_available_bath} qubits, J={J}, h={h}")
    
    for noise_idx, noise_factor in enumerate(noise_factors):
        single_qubit_noise = base_single_qubit_noise * noise_factor
        two_qubit_noise = base_two_qubit_noise * noise_factor
        
        # Construct filename
        vc_filename = f"variational_cooling_data_sys{largest_available_system}_bath{largest_available_bath}_J{J}_h{h}_sweeps{num_sweeps}_noise{single_qubit_noise}_{two_qubit_noise}.json"
        vc_filepath = os.path.join(results_dir, vc_filename)
        
        if not os.path.exists(vc_filepath):
            print(f"Warning: File not found for noise factor {noise_factor}: {vc_filename}")
            continue
        
        try:
            # Load data
            with open(vc_filepath, 'r') as f:
                vc_data = json.load(f)
            
            # Get measurements from final sweep
            final_sweep_key = f"sweep_{num_sweeps}"
            vc_measurements = vc_data['final_results']['measurements'][final_sweep_key]
            
            # Extract magnetization data
            positions = []
            magnetizations = []
            total_errors = []
            
            for i in range(largest_available_system):
                z_label = f"Z_{i}"
                if z_label in vc_measurements:
                    z_data = vc_measurements[z_label]
                    positions.append(i)
                    magnetizations.append(z_data['mean'])
                    total_errors.append(z_data['total_error'])
            
            if len(magnetizations) > 0:
                color = colors[noise_idx]
                
                # Plot magnetization with error bars
                ax.plot(positions, magnetizations, 'o-', 
                       color=color, linewidth=2, markersize=4, 
                       label=f'Noise {noise_factor:.1f}')
                
                # Add error bars
                ax.errorbar(positions, magnetizations, 
                           yerr=total_errors, fmt='none', 
                           capsize=2, capthick=1, ecolor=color, 
                           elinewidth=1, alpha=0.6)
                
                print(f"  Noise {noise_factor:.1f}: {len(magnetizations)} data points")
            else:
                print(f"  Noise {noise_factor:.1f}: No magnetization data found")
                
        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            print(f"Warning: Could not process data for noise factor {noise_factor}: {e}")
            continue
    
    # Customize plot
    ax.set_xlabel('Position i', fontsize=12)
    ax.set_ylabel('$\langle \sigma^Z_i \\rangle$', fontsize=12)
    ax.set_title(f'Magnetization vs Position for Different Noise Levels\nSystem: {largest_available_system}+{largest_available_bath} qubits, J={J}, h={h}', fontsize=14)
    ax.grid(True, alpha=0.3)
        
    # Set x-ticks to integers
    if largest_available_system:
        ax.set_xticks(range(largest_available_system))
    
    # Add colorbar for noise factor
    norm = plt.Normalize(0, 1)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8, aspect=30)
    cbar.set_label('Noise Factor', fontsize=12)
    cbar.set_ticks(np.linspace(0, 1, 6))  # Show 6 ticks from 0 to 1
    
    plt.tight_layout()
    
    # Save the plot
    filename = f"magnetization_vs_position_different_noise_J{J}_h{h}_sys{largest_available_system}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Magnetization vs position plot saved to: {filepath}")
    
    # Print summary
    print(f"\nMagnetization vs position analysis completed!")
    print(f"System size: {largest_available_system}+{largest_available_bath} qubits")
    print(f"Parameters: J={J}, h={h}")
    print(f"Noise levels: {len(noise_factors)} (0.0 to 1.0)")
    
    plt.show()


def plot_raw_spin_spin_correlations_different_jh_zero_noise(results_dir: str = "results", 
                                                           output_dir: str = "plots") -> None:
    """
    Plot raw spin-spin correlations <sigma^Z_i sigma^Z_j> vs distance for different J,h combinations at zero noise.
    Uses the largest available system size. Does NOT subtract individual expectations.
    
    Args:
        results_dir: directory containing results
        output_dir: directory to save plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define J,h combinations to plot
    J_h_combinations = [(0.6, 0.4), (0.55, 0.45), (0.45, 0.55), (0.4, 0.6)]
    
    # Define system sizes to check (largest first)
    system_sizes = [(28, 14), (24, 12), (20, 10), (16, 8), (12, 6), (8, 4), (4, 2)]
    
    # Fixed parameters
    num_sweeps = 12
    single_qubit_noise = 0.0
    two_qubit_noise = 0.0
    
    # Find the largest available system size with zero noise data
    largest_available_system = None
    largest_available_bath = None
    
    for system_qubits, bath_qubits in system_sizes:
        # Check if we have data for ALL J,h combinations
        all_combinations_available = True
        for J, h in J_h_combinations:
            vc_filename = f"variational_cooling_data_sys{system_qubits}_bath{bath_qubits}_J{J}_h{h}_sweeps{num_sweeps}_noise{single_qubit_noise}_{two_qubit_noise}.json"
            vc_filepath = os.path.join(results_dir, vc_filename)
            
            if not os.path.exists(vc_filepath):
                all_combinations_available = False
                break
        
        if all_combinations_available:
            largest_available_system = system_qubits
            largest_available_bath = bath_qubits
            break
    
    if largest_available_system is None:
        print("No zero noise data found for any system size!")
        return
    
    print(f"Using largest available system size: {largest_available_system}+{largest_available_bath} qubits")
    
    # Generate site pairs using the alternating center-outward pattern
    site_pairs = []
    center = largest_available_system // 2
    i, j = center, center
    
    while (i > 0) or (j < largest_available_system - 1):
        # Try to decrease i
        if i > 0:
            i -= 1
            if i < j:  # Only add if i < j to avoid duplicates
                site_pairs.append((i, j))
        
        # Try to increase j
        if j < largest_available_system - 1:
            j += 1
            if i < j:  # Only add if i < j to avoid duplicates
                site_pairs.append((i, j))
    
    # Calculate distances
    distances = [j - i for i, j in site_pairs]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Create colorbar based on J values (blues to reds)
    J_values = [combo[0] for combo in J_h_combinations]
    min_J, max_J = min(J_values), max(J_values)
    
    # Store magnetization data for inset
    magnetization_data = {}
    
    for combo_idx, (J, h) in enumerate(J_h_combinations):
        # Color based on J value (blues to reds)
        color = plt.cm.rainbow((J - min_J) / (max_J - min_J))
        
        # Check if variational cooling data exists
        vc_filename = f"variational_cooling_data_sys{largest_available_system}_bath{largest_available_bath}_J{J}_h{h}_sweeps{num_sweeps}_noise{single_qubit_noise}_{two_qubit_noise}.json"
        vc_filepath = os.path.join(results_dir, vc_filename)
        
        if not os.path.exists(vc_filepath):
            print(f"Warning: Variational cooling data not found for J={J}, h={h}")
            continue
        
        try:
            # Load variational cooling data
            with open(vc_filepath, 'r') as f:
                vc_data = json.load(f)
            
            # Get measurements from final sweep
            final_sweep_key = f"sweep_{num_sweeps}"
            vc_measurements = vc_data['final_results']['measurements'][final_sweep_key]
            
            # Calculate raw correlations (without subtracting individual expectations)
            raw_correlations = []
            valid_distances = []
            total_errors = []
            
            for i, j in site_pairs:
                # Get ZZ correlation
                zz_label = f"ZZ_{i}_{j}"
                if zz_label not in vc_measurements:
                    continue
                
                zz_data = vc_measurements[zz_label]
                raw_correlations.append(zz_data['mean'])
                valid_distances.append(j - i)
                total_errors.append(zz_data['total_error'])
            
            # Extract magnetization data for inset
            positions = []
            magnetizations = []
            magnetization_errors = []
            
            for i in range(largest_available_system):
                z_label = f"Z_{i}"
                if z_label in vc_measurements:
                    z_data = vc_measurements[z_label]
                    positions.append(i)
                    magnetizations.append(z_data['mean'])
                    magnetization_errors.append(z_data['total_error'])
            
            # Store magnetization data for this J,h combination
            magnetization_data[(J, h)] = {
                'positions': positions,
                'magnetizations': magnetizations,
                'errors': magnetization_errors,
                'color': color
            }
            
            if len(raw_correlations) > 0:
                # Plot raw correlations with error bars, add transparency for more sophisticated look
                ax.plot(valid_distances, raw_correlations, 'o-', 
                       color=color, linewidth=2, markersize=6, 
                       label=f'J={J}, h={h}')
                
                # Add error bars with transparency
                ax.errorbar(valid_distances, raw_correlations, 
                           yerr=total_errors, fmt='none', 
                           capsize=3, capthick=1, ecolor=color, 
                           elinewidth=1)
                
                print(f"  J={J}, h={h}: {len(raw_correlations)} data points")
            else:
                print(f"  J={J}, h={h}: No raw correlation data found")
                
        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            print(f"Warning: Could not process data for J={J}, h={h}: {e}")
            continue
    
    # Customize plot
    ax.set_xlabel('Distance |i-j|', fontsize=12)
    ax.set_ylabel('$\langle \sigma^Z_i \sigma^Z_j \\rangle$', fontsize=12)
    ax.set_title(f'Raw Spin-Spin Correlations vs Distance\nSystem: {largest_available_system}+{largest_available_bath} qubits, Zero Noise', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Move legend to top left, slightly to the left of where inset will be
    ax.legend(fontsize=10, loc='upper left', bbox_to_anchor=(0.2, 0.98))
    
    # Set x-ticks to integers
    ax.set_xticks(distances)
    
    # Create inset for magnetization vs position in top right corner
    if magnetization_data:
        # Create larger inset in top right corner, taking up more space
        axins = ax.inset_axes([0.515, 0.52, 0.47, 0.47])
        
        # Plot magnetization data for each J,h combination
        for (J, h), data in magnetization_data.items():
            positions = data['positions']
            magnetizations = data['magnetizations']
            errors = data['errors']
            color = data['color']
            
            if len(positions) > 0:
                # Plot magnetization with error bars, add transparency for more sophisticated look
                axins.plot(positions, magnetizations, 'o-', 
                          color=color, linewidth=2, markersize=4, 
                          label=f'J={J}, h={h}')
                
                # Add error bars with transparency
                axins.errorbar(positions, magnetizations, 
                              yerr=errors, fmt='none', 
                              capsize=3, capthick=1, ecolor=color, 
                              elinewidth=1)
        
        # Customize inset - remove title, increase font sizes
        axins.set_xlabel('Position i', fontsize=11)
        axins.set_ylabel('$\langle \sigma^Z_i \\rangle$', fontsize=11)
        axins.grid(True, alpha=0.3)
        
        # Set x-ticks to integers for inset
        if largest_available_system:
            axins.set_xticks(range(0, largest_available_system, max(1, largest_available_system//6)))
        
        # Remove inset legend as requested
    
    plt.tight_layout()
    
    # Save the plot
    filename = f"raw_spin_spin_correlations_different_jh_zero_noise_sys{largest_available_system}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Raw spin-spin correlations plot saved to: {filepath}")
    
    # Print summary
    print(f"\nRaw spin-spin correlations analysis completed!")
    print(f"System size: {largest_available_system}+{largest_available_bath} qubits")
    print(f"J,h combinations: {len(J_h_combinations)}")
    print(f"Zero noise analysis")
    
    plt.show()


def plot_raw_spin_spin_correlations_different_noise_fixed_jh(results_dir: str = "results", 
                                                           output_dir: str = "plots") -> None:
    """
    Plot raw spin-spin correlations <sigma^Z_i sigma^Z_j> vs distance for different noise levels at fixed J=0.6, h=0.4.
    Uses the largest available system size. Does NOT subtract individual expectations.
    
    Args:
        results_dir: directory containing results
        output_dir: directory to save plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Fixed parameters
    J = 0.6
    h = 0.4
    num_sweeps = 12
    
    # Define system sizes to check (largest first)
    system_sizes = [(28, 14), (24, 12), (20, 10), (16, 8), (12, 6), (8, 4), (4, 2)]
    
    # Define noise levels
    noise_factors = np.linspace(0, 1, 11)
    base_single_qubit_noise = 0.001
    base_two_qubit_noise = 0.01
    
    # Find the largest available system size
    largest_available_system = None
    largest_available_bath = None
    
    for system_qubits, bath_qubits in system_sizes:
        # Check if we have data for at least one noise level
        has_data = False
        for noise_factor in noise_factors:
            single_qubit_noise = base_single_qubit_noise * noise_factor
            two_qubit_noise = base_two_qubit_noise * noise_factor
            vc_filename = f"variational_cooling_data_sys{system_qubits}_bath{bath_qubits}_J{J}_h{h}_sweeps{num_sweeps}_noise{single_qubit_noise}_{two_qubit_noise}.json"
            vc_filepath = os.path.join(results_dir, vc_filename)
            
            if os.path.exists(vc_filepath):
                has_data = True
                break
        
        if has_data:
            largest_available_system = system_qubits
            largest_available_bath = bath_qubits
            break
    
    if largest_available_system is None:
        print("No data found for any system size!")
        return
    
    print(f"Using largest available system size: {largest_available_system}+{largest_available_bath} qubits")
    
    # Generate site pairs using the alternating center-outward pattern
    site_pairs = []
    center = largest_available_system // 2
    i, j = center, center
    
    while (i > 0) or (j < largest_available_system - 1):
        # Try to decrease i
        if i > 0:
            i -= 1
            if i < j:  # Only add if i < j to avoid duplicates
                site_pairs.append((i, j))
        
        # Try to increase j
        if j < largest_available_system - 1:
            j += 1
            if i < j:  # Only add if i < j to avoid duplicates
                site_pairs.append((i, j))
    
    # Calculate distances
    distances = [j - i for i, j in site_pairs]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Colors for different noise levels
    colors = plt.cm.viridis(np.linspace(0, 1, len(noise_factors)))
    
    # Store plateau values for inset
    plateau_values = []
    noise_levels_for_plateau = []
    
    print(f"Analyzing raw spin-spin correlations vs distance for different noise levels")
    print(f"System: {largest_available_system}+{largest_available_bath} qubits, J={J}, h={h}")
    
    for noise_idx, noise_factor in enumerate(noise_factors):
        single_qubit_noise = base_single_qubit_noise * noise_factor
        two_qubit_noise = base_two_qubit_noise * noise_factor
        
        # Construct filename
        vc_filename = f"variational_cooling_data_sys{largest_available_system}_bath{largest_available_bath}_J{J}_h{h}_sweeps{num_sweeps}_noise{single_qubit_noise}_{two_qubit_noise}.json"
        vc_filepath = os.path.join(results_dir, vc_filename)
        
        if not os.path.exists(vc_filepath):
            print(f"Warning: File not found for noise factor {noise_factor}: {vc_filename}")
            continue
        
        try:
            # Load data
            with open(vc_filepath, 'r') as f:
                vc_data = json.load(f)
            
            # Get measurements from final sweep
            final_sweep_key = f"sweep_{num_sweeps}"
            vc_measurements = vc_data['final_results']['measurements'][final_sweep_key]
            
            # Calculate raw correlations (without subtracting individual expectations)
            raw_correlations = []
            valid_distances = []
            total_errors = []
            
            for i, j in site_pairs:
                # Get ZZ correlation
                zz_label = f"ZZ_{i}_{j}"
                if zz_label not in vc_measurements:
                    continue
                
                zz_data = vc_measurements[zz_label]
                raw_correlations.append(zz_data['mean'])
                valid_distances.append(j - i)
                total_errors.append(zz_data['total_error'])
            
            if len(raw_correlations) > 0:
                color = colors[noise_idx]
                
                # Plot raw correlations with error bars
                ax.plot(valid_distances, raw_correlations, 'o-', 
                       color=color, linewidth=2, markersize=4, 
                       label=f'Noise {noise_factor:.1f}')
                
                # Add error bars
                ax.errorbar(valid_distances, raw_correlations, 
                           yerr=total_errors, fmt='none', 
                           capsize=2, capthick=1, ecolor=color, 
                           elinewidth=1, alpha=0.6)
                
                # Analyze plateau value
                # Look for plateau in the specific distance range |i-j| = 12 to 18
                plateau_mask = (np.array(valid_distances) >= 12) & (np.array(valid_distances) <= 18)
                
                if np.any(plateau_mask):
                    plateau_correlations = np.array(raw_correlations)[plateau_mask]
                    plateau_distances = np.array(valid_distances)[plateau_mask]
                    plateau_value = np.mean(plateau_correlations)
                    plateau_values.append(plateau_value)
                    noise_levels_for_plateau.append(noise_factor)
                    
                    # Add horizontal dotted line at plateau value
                    ax.axhline(y=plateau_value, color=color, linestyle=':', alpha=0.5, linewidth=1)
                    
                    print(f"  Noise {noise_factor:.1f}: {len(raw_correlations)} data points, plateau value = {plateau_value:.4f} (distances {plateau_distances})")
                else:
                    print(f"  Noise {noise_factor:.1f}: {len(raw_correlations)} data points, no plateau detected in range |i-j| = 12-18")
            else:
                print(f"  Noise {noise_factor:.1f}: No raw correlation data found")
                
        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            print(f"Warning: Could not process data for noise factor {noise_factor}: {e}")
            continue
    
    # Customize plot
    ax.set_xlabel('Distance |i-j|', fontsize=12)
    ax.set_ylabel('$\langle \sigma^Z_i \sigma^Z_j \\rangle$', fontsize=12)
    ax.set_title(f'Raw Spin-Spin Correlations vs Distance for Different Noise Levels\nSystem: {largest_available_system}+{largest_available_bath} qubits, J={J}, h={h}', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Set x-ticks to integers
    ax.set_xticks(distances)
    
    # Add colorbar for noise factor
    norm = plt.Normalize(0, 1)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8, aspect=30)
    cbar.set_label('Noise Factor', fontsize=12)
    cbar.set_ticks(np.linspace(0, 1, 6))  # Show 6 ticks from 0 to 1
    
    # Create inset for plateau values vs noise
    if len(plateau_values) > 1:
        # Create inset in the upper right corner
        axins = ax.inset_axes([0.6, 0.6, 0.35, 0.35])
        
        # Sort by noise level
        sorted_indices = np.argsort(noise_levels_for_plateau)
        sorted_noise = [noise_levels_for_plateau[i] for i in sorted_indices]
        sorted_plateau = [plateau_values[i] for i in sorted_indices]

        # Connect points with line
        axins.plot(sorted_noise, sorted_plateau, '-', color='red', linewidth=2, alpha=0.7)
        
        # Plot plateau values vs noise with matching colors
        for i, (noise, plateau) in enumerate(zip(sorted_noise, sorted_plateau)):
            color = plt.cm.viridis(noise)  # Use noise factor as color
            axins.plot(noise, plateau, 'o', color=color, markersize=6)

        axins.set_xlabel('Noise Factor', fontsize=10)
        axins.set_ylabel('Plateau Value', fontsize=10)
        axins.set_title('Plateau Value vs Noise', fontsize=11)
        axins.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    filename = f"raw_spin_spin_correlations_different_noise_J{J}_h{h}_sys{largest_available_system}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Raw spin-spin correlations plot saved to: {filepath}")
    
    # Print summary
    print(f"\nRaw spin-spin correlations analysis completed!")
    print(f"System size: {largest_available_system}+{largest_available_bath} qubits")
    print(f"Parameters: J={J}, h={h}")
    print(f"Noise levels: {len(noise_factors)} (0.0 to 1.0)")
    print(f"Plateau values found: {len(plateau_values)}/{len(noise_factors)} noise levels")
    
    plt.show()


def create_noise_analysis_plot(results_dir: str = "results", plots_dir: str = "plots") -> None:
    """
    Standalone function to create the energy density vs noise plot for different system sizes.
    
    Args:
        results_dir: directory containing results
        plots_dir: directory to save plots
    """
    print("Creating energy density vs noise plot for different system sizes...")
    plot_energy_density_vs_noise_for_different_system_sizes(results_dir, plots_dir)
    
    print("Creating correlation length vs noise plot...")
    plot_correlation_length_vs_noise_largest_system(results_dir, plots_dir)
    
    print("Creating spin-spin correlations plot for largest system with zero noise...")
    plot_spin_spin_correlations_largest_system_zero_noise(results_dir, plots_dir)
    
    print("Creating magnetization vs position plot for different J,h combinations at zero noise...")
    plot_magnetization_vs_position_different_jh_zero_noise(results_dir, plots_dir)
    
    print("Creating magnetization vs position plot for different noise levels...")
    plot_magnetization_vs_position_different_noise_fixed_jh(results_dir, plots_dir)
    
    print("Creating raw spin-spin correlations plot for different J,h combinations at zero noise...")
    plot_raw_spin_spin_correlations_different_jh_zero_noise(results_dir, plots_dir)
    
    print("Creating raw spin-spin correlations plot for different noise levels...")
    plot_raw_spin_spin_correlations_different_noise_fixed_jh(results_dir, plots_dir)
    
    print("Noise analysis plot completed!")


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
    
    print("Creating energy density vs noise plot for different system sizes...")
    plot_energy_density_vs_noise_for_different_system_sizes(results_dir, plots_dir)
    
    print("Creating correlation length vs noise plot...")
    plot_correlation_length_vs_noise_largest_system(results_dir, plots_dir)
    
    print("Creating spin-spin correlations plot for largest system with zero noise...")
    plot_spin_spin_correlations_largest_system_zero_noise(results_dir, plots_dir)
    
    print("Creating magnetization vs position plot for different J,h combinations at zero noise...")
    plot_magnetization_vs_position_different_jh_zero_noise(results_dir, plots_dir)
    
    print("Creating magnetization vs position plot for different noise levels...")
    plot_magnetization_vs_position_different_noise_fixed_jh(results_dir, plots_dir)
    
    print("Creating raw spin-spin correlations plot for different J,h combinations at zero noise...")
    plot_raw_spin_spin_correlations_different_jh_zero_noise(results_dir, plots_dir)
    
    print("Creating raw spin-spin correlations plot for different noise levels...")
    plot_raw_spin_spin_correlations_different_noise_fixed_jh(results_dir, plots_dir)
    
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
    parser.add_argument('--noise_plot_only', action='store_true', 
                       help='Only create the energy density vs noise plot for different system sizes')
    
    args = parser.parse_args()
    
    if args.noise_plot_only:
        # Only create the noise plot
        create_noise_analysis_plot(
            results_dir=args.results_dir,
            plots_dir=args.plots_dir
        )
    else:
        # Run full analysis
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
            'system_qubits': 28,
            'bath_qubits': 14,
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
        
        # Also create the energy density vs noise plot for different system sizes
        print("\nCreating comprehensive analysis plots...")
        create_noise_analysis_plot(
            results_dir=params['results_dir'], 
            plots_dir=params['plots_dir']
        )
