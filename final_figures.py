#!/usr/bin/env python3
"""
Final figures generation script for variational cooling MPS simulation.
Creates publication-ready plots with consistent styling and formatting.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from typing import List, Tuple, Dict, Any, Optional
import argparse
from matplotlib.lines import Line2D


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


def get_marker_for_jh(J, h):
    """Get consistent marker for J,h combination across all plots"""
    if (J, h) == (0.6, 0.4):
        return 'o'  # Circle
    elif (J, h) == (0.55, 0.45):
        return 's'  # Square
    elif (J, h) == (0.45, 0.55):
        return '^'  # Triangle
    elif (J, h) == (0.4, 0.6):
        return 'D'  # Diamond
    else:
        return 'o'  # Default to circle


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


def plot_energy_density_vs_two_qubit_noise(results_dir: str = "results", 
                                          output_dir: str = "plots/final",
                                          J: float = 0.6, h: float = 0.4,
                                          marker_alpha: float = 0.8, line_alpha: float = 0.6) -> None:
    """
    Plot energy density above ground state vs two-qubit gate error rate for different system sizes.
    
    Args:
        results_dir: directory containing results
        output_dir: directory to save plots
        J: Ising coupling strength
        h: transverse field strength
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define system sizes to plot (only 4, 8, 16, 28 system qubits)
    system_sizes = [(4, 2), (8, 4), (16, 8), (28, 14)]
    
    # Define noise levels
    noise_factors = np.linspace(0, 1, 11)
    base_single_qubit_noise = 0.001
    base_two_qubit_noise = 0.01
    
    # Fixed parameters
    num_sweeps = 12
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Define line styles for different system sizes (largest gets solid line)
    line_styles = [':','-.','--','-']  # 4, 8, 16, 28 qubits
    
    print(f"Creating energy density vs two-qubit noise plot for J={J}, h={h}")
    
    # Store all data for plotting with colormap
    all_data = []
    
    for sys_idx, (system_qubits, bath_qubits) in enumerate(system_sizes):
        print(f"  Processing system size: {system_qubits}+{bath_qubits} qubits")
        two_qubit_noise_levels = []
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
                            
                            two_qubit_noise_levels.append(two_qubit_noise)
                            energy_densities.append(energy_density_above_gs)
                            total_errors.append(total_err)
                        except (json.JSONDecodeError, KeyError) as e:
                            print(f"    Warning: Could not process ground state file {gs_filename}: {e}")
                            continue
                    else:
                        print(f"    Warning: Ground state file not found: {gs_filename}")
                        continue
                    
                except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
                    print(f"    Warning: Could not process {filename}: {e}")
                    continue
        
        if two_qubit_noise_levels:  # Only plot if we have data
            print(f"    Found {len(two_qubit_noise_levels)} data points")
            
            # Sort by two-qubit noise level
            sorted_indices = np.argsort(two_qubit_noise_levels)
            two_qubit_noise_levels = [two_qubit_noise_levels[i] for i in sorted_indices]
            energy_densities = [energy_densities[i] for i in sorted_indices]
            total_errors = [total_errors[i] for i in sorted_indices]
            
            # Store data for this system size
            all_data.append({
                'system_qubits': system_qubits,
                'linestyle': line_styles[sys_idx],
                'two_qubit_noise_levels': two_qubit_noise_levels,
                'energy_densities': energy_densities,
                'total_errors': total_errors
            })
        else:
            print(f"    No data found for this system size")
    
    # Plot all data with colormap based on two-qubit noise level
    for data in all_data:
        system_qubits = data['system_qubits']
        linestyle = data['linestyle']
        two_qubit_noise_levels = data['two_qubit_noise_levels']
        energy_densities = data['energy_densities']
        total_errors = data['total_errors']
        
        # Get marker for this J,h combination
        marker = get_marker_for_jh(J, h)
        
        # Add line connecting points with appropriate linestyle
        ax.plot(two_qubit_noise_levels, energy_densities, 
               color='gray', linestyle=linestyle, linewidth=1.5, alpha=line_alpha, zorder=1)
        
        # Add error bars with colors matching the markers
        for i, (x, y, err) in enumerate(zip(two_qubit_noise_levels, energy_densities, total_errors)):
            color = plt.cm.viridis(x / 0.01)  # Scale to 0-1 range for colormap
            ax.errorbar([x], [y], yerr=[err], fmt='none', 
                       capsize=4, capthick=1.5, ecolor=color, 
                       elinewidth=1.5, alpha=marker_alpha, zorder=2)
        
        # Create scatter plot with colormap based on two-qubit noise level and J,h marker
        scatter = ax.scatter(two_qubit_noise_levels, energy_densities, 
                           c=two_qubit_noise_levels, cmap='viridis', 
                           s=60, alpha=marker_alpha, marker=marker,
                           label=f'{system_qubits} + {system_qubits//2} qubits', zorder=3)
    
    # # Add colorbar for two-qubit gate error rate
    # norm = plt.Normalize(0, 0.01)
    # sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
    # sm.set_array([])
    # cbar = plt.colorbar(sm, ax=ax, shrink=0.8, aspect=30)
    # cbar.set_label('Two-Qubit Gate Error Rate', fontsize=12)
    # cbar.set_ticks(np.linspace(0, 0.01, 6))  # Show 6 ticks from 0 to 0.01
    
    # Create custom legend for system sizes
    legend_elements = []
    for i, data in enumerate(all_data):
        system_qubits = data['system_qubits']
        linestyle = data['linestyle']
        legend_elements.append(Line2D([0], [0], color='gray', linestyle=linestyle, 
                                    linewidth=1.5, label=f'{system_qubits} + {system_qubits//2} qubits'))
    
    # Customize plot
    ax.set_xlabel('Two-Qubit Gate Error Rate', fontsize=14)
    ax.set_ylabel('Energy Density Above Ground State', fontsize=14)
    ax.set_title(f'Energy Density vs Two-Qubit Gate Error Rate\nJ={J}, h={h}', fontsize=16)
    ax.grid(False) # ax.grid(True, alpha=0.3)
    ax.legend(handles=legend_elements, fontsize=12, loc='upper left')
    
    # Set y-axis to start at 0
    ax.set_ylim(bottom=0)
    
    # Set x-ticks to show two-qubit noise levels clearly
    ax.set_xticks(np.linspace(0, 0.01, 6))
    
    plt.tight_layout()
    
    # Save the plot
    filename = f"energy_density_vs_two_qubit_noise_J{J}_h{h}"
    filepath_pdf = os.path.join(output_dir, filename+".pdf")
    filepath_png = os.path.join(output_dir, filename+".png")
    plt.savefig(filepath_pdf, dpi=300, bbox_inches='tight')
    plt.savefig(filepath_png, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_raw_spin_spin_correlations_different_jh_zero_noise_no_inset(results_dir: str = "results", 
                                                                   output_dir: str = "plots/final",
                                                                   marker_alpha: float = 0.8, line_alpha: float = 0.6) -> None:
    """
    Plot raw spin-spin correlations <sigma^Z_i sigma^Z_j> vs distance for different J,h combinations at zero noise.
    Uses the largest available system size. Does NOT subtract individual expectations.
    No inset included.
    
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
    
    # Use viridis colormap for consistent coloring (all at zero noise = 0.0)
    colors = [plt.cm.viridis(0.0) for _ in J_h_combinations]  # All same color since noise = 0
    
    for combo_idx, (J, h) in enumerate(J_h_combinations):
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
            
            if len(raw_correlations) > 0:
                color = colors[combo_idx]
                marker = get_marker_for_jh(J, h)
                
                # Plot raw correlations with error bars
                ax.plot(valid_distances, raw_correlations, 
                       color=color, marker=marker, linewidth=1.5, 
                       markersize=8, alpha=line_alpha, label=f'J={J}, h={h}')
                
                # Add error bars with matching colors
                ax.errorbar(valid_distances, raw_correlations, 
                           yerr=total_errors, fmt='none', 
                           capsize=4, capthick=1.5, ecolor=color, 
                           elinewidth=1.5, alpha=marker_alpha)
                
                print(f"  J={J}, h={h}: {len(raw_correlations)} data points")
            else:
                print(f"  J={J}, h={h}: No raw correlation data found")
                
        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            print(f"Warning: Could not process data for J={J}, h={h}: {e}")
            continue
    
    # Customize plot
    ax.set_xlabel('Distance |i-j|', fontsize=14)
    ax.set_ylabel('$\langle \sigma^Z_i \sigma^Z_j \\rangle$', fontsize=14)
    ax.set_title(f'Raw Spin-Spin Correlations vs Distance\nSystem: {largest_available_system}+{largest_available_bath} qubits, Zero Noise', fontsize=16)
    ax.grid(False) # ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='upper right')
    
    # Set x-ticks to integers
    ax.set_xticks(distances)
    
    # # Add colorbar for two-qubit gate error rate (even though all data is at zero noise)
    # norm = plt.Normalize(0, 0.01)
    # sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
    # sm.set_array([])
    # cbar = plt.colorbar(sm, ax=ax, shrink=0.8, aspect=30)
    # cbar.set_label('Two-Qubit Gate Error Rate', fontsize=12)
    # cbar.set_ticks(np.linspace(0, 0.01, 6))  # Show 6 ticks from 0 to 0.01
    
    plt.tight_layout()
    
    # Save the plot
    filename = f"raw_spin_spin_correlations_different_jh_zero_noise_no_inset_sys{largest_available_system}"
    filepath_pdf = os.path.join(output_dir, filename+".pdf")
    filepath_png = os.path.join(output_dir, filename+".png")
    plt.savefig(filepath_pdf, dpi=300, bbox_inches='tight')
    plt.savefig(filepath_png, dpi=300, bbox_inches='tight')
    print(f"Raw spin-spin correlations plot saved to: {filepath_pdf}")
    print(f"Raw spin-spin correlations plot saved to: {filepath_png}")
    
    plt.show()


def plot_raw_spin_spin_correlations_different_noise_no_red_line(results_dir: str = "results", 
                                                              output_dir: str = "plots/final",
                                                              marker_alpha: float = 0.8, line_alpha: float = 0.6) -> None:
    """
    Plot raw spin-spin correlations <sigma^Z_i sigma^Z_j> vs distance for different noise levels at fixed J=0.6, h=0.4.
    Uses the largest available system size. Does NOT subtract individual expectations.
    Remove the red line from the inset.
    
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
    
    # Get marker for this J,h combination (J=0.6, h=0.4)
    marker = get_marker_for_jh(J, h)
    
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
                
                # Plot raw correlations with error bars using consistent marker
                ax.plot(valid_distances, raw_correlations, 
                       color=color, marker=marker, linewidth=1.5, markersize=4, 
                       alpha=line_alpha, label=f'Noise {noise_factor:.1f}')
                
                # Add error bars with matching colors
                ax.errorbar(valid_distances, raw_correlations, 
                           yerr=total_errors, fmt='none', 
                           capsize=2, capthick=1, ecolor=color, 
                           elinewidth=1.5, alpha=marker_alpha)
                
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
    ax.set_xlabel('Distance |i-j|', fontsize=14)
    ax.set_ylabel('$\langle \sigma^Z_i \sigma^Z_j \\rangle$', fontsize=14)
    ax.set_title(f'Raw Spin-Spin Correlations vs Distance for Different Noise Levels\nSystem: {largest_available_system}+{largest_available_bath} qubits, J={J}, h={h}', fontsize=16)
    ax.grid(False) # ax.grid(True, alpha=0.3)
    
    # Set x-ticks to integers
    ax.set_xticks(distances)
    
    # Add colorbar for two-qubit gate error rate
    norm = plt.Normalize(0, 0.01)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8, aspect=30)
    cbar.set_label('Two-Qubit Gate Error Rate', fontsize=12)
    cbar.set_ticks(np.linspace(0, 0.01, 6))  # Show 6 ticks from 0 to 0.01
    
    # Create inset for plateau values vs noise
    if len(plateau_values) > 1:
        # Create inset in the upper right corner
        axins = ax.inset_axes([0.6, 0.6, 0.35, 0.35])
        
        # Sort by noise level
        sorted_indices = np.argsort(noise_levels_for_plateau)
        sorted_noise = [noise_levels_for_plateau[i] for i in sorted_indices]
        sorted_plateau = [plateau_values[i] for i in sorted_indices]

        # Plot plateau values vs noise with matching colors (NO red line)
        for i, (noise, plateau) in enumerate(zip(sorted_noise, sorted_plateau)):
            # Convert noise factor to actual two-qubit gate error rate
            two_qubit_error_rate = noise * 0.01  # noise factor * base_two_qubit_noise
            color = plt.cm.viridis(two_qubit_error_rate / 0.01)  # Scale to 0-1 range for viridis colormap
            axins.plot(two_qubit_error_rate, plateau, marker, color=color, markersize=6, alpha=marker_alpha)

        axins.set_xlabel('Two-Qubit Gate Error Rate', fontsize=10)
        axins.set_ylabel('Plateau Value', fontsize=10)
        axins.set_title('Plateau Value vs Noise', fontsize=11)
        axins.grid(False) # axins.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    filename = f"raw_spin_spin_correlations_different_noise_J{J}_h{h}_sys{largest_available_system}"
    filepath_pdf = os.path.join(output_dir, filename+".pdf")
    filepath_png = os.path.join(output_dir, filename+".png")
    plt.savefig(filepath_pdf, dpi=300, bbox_inches='tight')
    plt.savefig(filepath_png, dpi=300, bbox_inches='tight')
    plt.show()


def create_final_figures(results_dir: str = "results", output_dir: str = "plots/final",
                        marker_alpha: float = 1., line_alpha: float = 1.) -> None:
    """
    Create all final figures for publication.
    
    Args:
        results_dir: directory containing results
        output_dir: directory to save plots
    """
    print("Creating final figures for publication...")
    print("=" * 50)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Energy density vs two-qubit noise for J=0.6, h=0.4
    print("\n1. Creating energy density vs two-qubit noise plot (J=0.6, h=0.4)...")
    plot_energy_density_vs_two_qubit_noise(results_dir, output_dir, J=0.6, h=0.4, 
                                          marker_alpha=marker_alpha, line_alpha=line_alpha)
    
    # 2. Energy density vs two-qubit noise for J=0.4, h=0.6
    print("\n2. Creating energy density vs two-qubit noise plot (J=0.4, h=0.6)...")
    plot_energy_density_vs_two_qubit_noise(results_dir, output_dir, J=0.4, h=0.6,
                                          marker_alpha=marker_alpha, line_alpha=line_alpha)
    
    # 3. Raw spin-spin correlations for different J,h at zero noise (no inset)
    print("\n3. Creating raw spin-spin correlations plot for different J,h (no inset)...")
    plot_raw_spin_spin_correlations_different_jh_zero_noise_no_inset(results_dir, output_dir,
                                                                    marker_alpha=marker_alpha, line_alpha=line_alpha)
    
    # 4. Raw spin-spin correlations for different noise levels (no red line in inset)
    print("\n4. Creating raw spin-spin correlations plot for different noise levels (no red line)...")
    plot_raw_spin_spin_correlations_different_noise_no_red_line(results_dir, output_dir,
                                                               marker_alpha=marker_alpha, line_alpha=line_alpha)
    
    print("\n" + "=" * 50)
    print("All final figures created successfully!")
    print(f"Figures saved to: {output_dir}")


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description='Create final figures for variational cooling data')
    
    parser.add_argument('--results_dir', type=str, default='results', help='Results directory')
    parser.add_argument('--output_dir', type=str, default='plots/final', help='Output directory for plots')
    parser.add_argument('--marker_alpha', type=float, default=1., help='Alpha value for markers and error bars')
    parser.add_argument('--line_alpha', type=float, default=1., help='Alpha value for lines')
    
    args = parser.parse_args()
    
    # Create all final figures
    create_final_figures(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        marker_alpha=args.marker_alpha,
        line_alpha=args.line_alpha
    )


if __name__ == "__main__":
    # Check if command line arguments were provided
    import sys
    if len(sys.argv) > 1:
        # Use command line arguments
        main()
    else:
        # Run with default parameters
        print("Final Figures Generation for Variational Cooling")
        print("=" * 50)
        
        # Create all final figures with default alpha values
        create_final_figures(marker_alpha=1., line_alpha=1.)
