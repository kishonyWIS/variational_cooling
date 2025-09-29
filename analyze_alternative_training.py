#!/usr/bin/env python3
"""
Analysis script for alternative training methods results.
Creates plots of steady state energy density vs system size for different training methods.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from typing import List, Tuple, Dict, Any, Optional
import argparse
from matplotlib.lines import Line2D
from cluster_job_generator_data_collection import generate_variational_cooling_filename


NUM_SWEEPS_STEADY_STATE = 40
MAX_NUM_SWEEPS = 40


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


def find_variational_cooling_file(results_dir, system_qubits, bath_qubits, J, h, num_sweeps, 
                                 single_qubit_gate_noise, two_qubit_gate_noise, training_method):
    """
    Find variational cooling data file, handling floating-point precision issues.
    
    Args:
        results_dir: directory containing results
        system_qubits: number of system qubits
        bath_qubits: number of bath qubits
        J: Ising coupling strength
        h: transverse field strength
        num_sweeps: number of sweeps
        single_qubit_gate_noise: single qubit gate noise level
        two_qubit_gate_noise: two qubit gate noise level
        training_method: training method name
    
    Returns:
        str or None: filepath if found, None otherwise
    """
    # Generate clean filename using shared function
    clean_filename = generate_variational_cooling_filename(
        system_qubits, bath_qubits, J, h, num_sweeps,
        single_qubit_gate_noise, two_qubit_gate_noise, training_method
    )
    
    # Also generate the potentially problematic filename (for backward compatibility)
    problematic_filename = f"variational_cooling_data_sys{system_qubits}_bath{bath_qubits}_J{J}_h{h}_sweeps{num_sweeps}_noise{single_qubit_gate_noise}_{two_qubit_gate_noise}_method{training_method}.json"
    
    # Check both filenames
    clean_filepath = os.path.join(results_dir, clean_filename)
    problematic_filepath = os.path.join(results_dir, problematic_filename)
    
    if os.path.exists(clean_filepath):
        return clean_filepath
    elif os.path.exists(problematic_filepath):
        return problematic_filepath
    else:
        return None


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


def get_training_method_color(training_method: str) -> str:
    """Get consistent color for training method across all plots"""
    colors = {
        'energy': 'blue',
        'pruning': 'red', 
        'random_initialization': 'green',
        'reoptimize_different_states': 'orange'
    }
    return colors.get(training_method, 'black')


def get_training_method_marker(training_method: str) -> str:
    """Get consistent marker for training method across all plots"""
    markers = {
        'energy': 'o',
        'pruning': 's',
        'random_initialization': '^',
        'reoptimize_different_states': 'D'
    }
    return markers.get(training_method, 'o')


def plot_energy_density_vs_system_size(results_dir: str = "results", 
                                      output_dir: str = "plots/alternative_analysis",
                                      J: float = 0.4, h: float = 0.6,
                                      training_methods: List[str] = None,
                                      marker_alpha: float = 0.8, line_alpha: float = 0.6,
                                      linewidth: float = 1.5, markersize: float = 8,
                                      ax: Optional[plt.Axes] = None) -> None:
    """
    Plot energy density above ground state vs system size for different training methods.
    
    Args:
        results_dir: directory containing results
        output_dir: directory to save plots
        J: Ising coupling strength
        h: transverse field strength
        training_methods: list of training methods to plot
        marker_alpha: alpha value for markers
        line_alpha: alpha value for lines
        linewidth: line width
        markersize: marker size
    """
    if training_methods is None:
        training_methods = ['energy', 'pruning', 'random_initialization', 'reoptimize_different_states']
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define system sizes to check (from cluster_job_generator_data_collection.py)
    system_sizes = [(4, 2), (8, 4), (12, 6), (16, 8), (20, 10), (24, 12), (28, 14)]
    
    single_qubit_noise = 0.0
    two_qubit_noise = 0.0
    
    # Create figure if no axis provided
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        create_new_figure = True
    else:
        fig = ax.figure
        create_new_figure = False
    
    print(f"Creating energy density above ground state vs system size plot for J={J}, h={h}")
    
    # Plot each training method
    for training_method in training_methods:
        print(f"  Processing training method: {training_method}")
        
        system_qubit_counts = []
        energy_densities = []
        total_errors = []
        
        # Collect data for each system size
        for system_qubits, bath_qubits in system_sizes:
            # Find file using robust filename matching
            filepath = find_variational_cooling_file(results_dir, system_qubits, bath_qubits, J, h, 
                                                   MAX_NUM_SWEEPS, single_qubit_noise, two_qubit_noise, training_method)
            
            if filepath:
                try:
                    # Load data
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    
                    # Get measurements from final sweep
                    final_sweep_key = f"sweep_{NUM_SWEEPS_STEADY_STATE}"
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
                            
                            system_qubit_counts.append(system_qubits)
                            energy_densities.append(energy_density_above_gs)
                            total_errors.append(total_err)
                            
                        except (json.JSONDecodeError, KeyError) as e:
                            print(f"    Warning: Could not process ground state file {gs_filename}: {e}")
                            continue
                    else:
                        print(f"    Warning: Ground state file not found: {gs_filename}")
                        continue
                    
                    print(f"    {system_qubits}+{bath_qubits} qubits: energy density above GS = {energy_density_above_gs:.6f} ± {total_err:.6f}")
                    
                except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
                    print(f"    Warning: Could not process {os.path.basename(filepath)}: {e}")
                    continue
            else:
                print(f"    Warning: File not found for system {system_qubits}+{bath_qubits}")
                continue
        
        if system_qubit_counts:  # Only plot if we have data
            print(f"    Found {len(system_qubit_counts)} data points for {training_method}")
            
            # Sort by system size
            sorted_indices = np.argsort(system_qubit_counts)
            system_qubit_counts = [system_qubit_counts[i] for i in sorted_indices]
            energy_densities = [energy_densities[i] for i in sorted_indices]
            total_errors = [total_errors[i] for i in sorted_indices]
            
            # Get color and marker for this training method
            color = get_training_method_color(training_method)
            marker = get_training_method_marker(training_method)
            
            # Plot with error bars
            legend_label = training_method if training_method != 'energy' else 'original'
            ax.errorbar(system_qubit_counts, energy_densities, yerr=total_errors,
                       color=color, marker=marker, linewidth=linewidth, markersize=markersize,
                       alpha=line_alpha, label=legend_label, capsize=4, capthick=linewidth)
        else:
            print(f"    No data found for training method: {training_method}")
    
    # Customize plot
    ax.set_xlabel('System Qubits', fontsize=14)
    ax.set_ylabel('Energy Density Above Ground State', fontsize=14)
    ax.set_title(f'Energy Density Above Ground State vs System Size\nJ={J}, h={h}', fontsize=16)
    # make y axis start at 0
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=12, loc='best')
    
    # Set x-ticks to system sizes
    ax.set_xticks([size[0] for size in system_sizes])
    
    # Only save and show if this is a new figure
    if create_new_figure:
        plt.tight_layout()
        
        # Save the plot
        filename = f"energy_density_vs_system_size_J{J}_h{h}"
        filepath_pdf = os.path.join(output_dir, filename+".pdf")
        filepath_png = os.path.join(output_dir, filename+".png")
        plt.savefig(filepath_pdf, dpi=300, bbox_inches='tight')
        plt.savefig(filepath_png, dpi=300, bbox_inches='tight')
        
        print(f"Plot saved to: {filepath_pdf}")
        print(f"Plot saved to: {filepath_png}")
        
        plt.show()


def plot_energy_density_vs_sweeps(results_dir: str = "results", 
                                 output_dir: str = "plots/alternative_analysis",
                                 J: float = 0.4, h: float = 0.6,
                                 system_qubits: int = 28, bath_qubits: int = 14,
                                 training_methods: List[str] = None,
                                 marker_alpha: float = 0.8, line_alpha: float = 0.6,
                                 linewidth: float = 1.5, markersize: float = 8) -> None:
    """
    Plot energy density above ground state vs sweeps for different training methods.
    
    Args:
        results_dir: directory containing results
        output_dir: directory to save plots
        J: Ising coupling strength
        h: transverse field strength
        system_qubits: number of system qubits
        bath_qubits: number of bath qubits
        training_methods: list of training methods to plot
        marker_alpha: alpha value for markers
        line_alpha: alpha value for lines
        linewidth: line width
        markersize: marker size
    """
    if training_methods is None:
        training_methods = ['energy', 'pruning', 'random_initialization', 'reoptimize_different_states']
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    single_qubit_noise = 0.0
    two_qubit_noise = 0.0
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    print(f"Creating energy density above ground state vs sweeps plot for J={J}, h={h}, system={system_qubits}+{bath_qubits}")
    
    # Get ground state energy for this system size
    gs_filename = f"ground_state_data_sys{system_qubits}_J{J}_h{h}.json"
    gs_filepath = os.path.join(results_dir, gs_filename)
    
    ground_state_energy_density = None
    if os.path.exists(gs_filepath):
        try:
            with open(gs_filepath, 'r') as f:
                gs_data = json.load(f)
            ground_state_energy_density = gs_data['ground_state_results']['ground_state_energy'] / system_qubits
            print(f"Ground state energy density: {ground_state_energy_density:.6f}")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not process ground state file {gs_filename}: {e}")
    else:
        print(f"Warning: Ground state file not found: {gs_filename}")
        return
    
    # Plot each training method
    for training_method in training_methods:
        print(f"  Processing training method: {training_method}")
        
        # Find file using robust filename matching
        filepath = find_variational_cooling_file(results_dir, system_qubits, bath_qubits, J, h, 
                                               MAX_NUM_SWEEPS, single_qubit_noise, two_qubit_noise, training_method)
        
        if filepath:
            try:
                # Load data
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                sweeps = []
                energy_densities = []
                total_errors = []
                
                # Collect data for each sweep (0 to num_sweeps)
                for sweep in range(NUM_SWEEPS_STEADY_STATE + 1):
                    sweep_key = f"sweep_{sweep}"
                    if sweep_key in data['final_results']['measurements']:
                        measurements = data['final_results']['measurements'][sweep_key]
                        
                        # Calculate energy density
                        energy_density, total_err = calculate_energy_density(
                            measurements, J, h, system_qubits
                        )
                        
                        # Calculate energy density above ground state
                        energy_density_above_gs = energy_density - ground_state_energy_density
                        
                        sweeps.append(sweep)
                        energy_densities.append(energy_density_above_gs)
                        total_errors.append(total_err)
                        
                        print(f"    Sweep {sweep}: energy density above GS = {energy_density_above_gs:.6f} ± {total_err:.6f}")
                
                if sweeps:  # Only plot if we have data
                    print(f"    Found {len(sweeps)} data points for {training_method}")
                    
                    # Get color and marker for this training method
                    color = get_training_method_color(training_method)
                    marker = get_training_method_marker(training_method)
                    
                    # Plot with error bars
                    legend_label = training_method if training_method != 'energy' else 'original'
                    ax.errorbar(sweeps, energy_densities, yerr=total_errors,
                               color=color, marker=marker, linewidth=linewidth, markersize=markersize,
                               alpha=line_alpha, label=legend_label, capsize=4, capthick=linewidth)
                else:
                    print(f"    No sweep data found for training method: {training_method}")
                    
            except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
                print(f"    Warning: Could not process {os.path.basename(filepath)}: {e}")
                continue
        else:
            print(f"    Warning: File not found for training method {training_method}")
            continue
    
    # Customize plot
    ax.set_xlabel('Sweep Number', fontsize=14)
    ax.set_ylabel('Energy Density Above Ground State', fontsize=14)
    ax.set_title(f'Energy Density Above Ground State vs Sweeps\nJ={J}, h={h}, System: {system_qubits}+{bath_qubits} qubits', fontsize=16)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Set x-ticks to integers
    ax.set_xticks(range(NUM_SWEEPS_STEADY_STATE + 1))
    
    # Save the plot
    plt.tight_layout()
    filename = f"energy_density_vs_sweeps_J{J}_h{h}_sys{system_qubits}"
    filepath_pdf = os.path.join(output_dir, filename+".pdf")
    filepath_png = os.path.join(output_dir, filename+".png")
    plt.savefig(filepath_pdf, dpi=300, bbox_inches='tight')
    plt.savefig(filepath_png, dpi=300, bbox_inches='tight')
    
    print(f"Plot saved to: {filepath_pdf}")
    print(f"Plot saved to: {filepath_png}")
    
    plt.show()


def plot_energy_density_vs_system_size_two_panels(results_dir: str = "results", 
                                                 output_dir: str = "plots/alternative_analysis",
                                                 training_methods: List[str] = None,
                                                 marker_alpha: float = 0.8, line_alpha: float = 0.6,
                                                 linewidth: float = 1.5, markersize: float = 8) -> None:
    """
    Plot energy density above ground state vs system size for different training methods
    in two panels (J=0.4,h=0.6 and J=0.45,h=0.55) with shared y-axis.
    
    Args:
        results_dir: directory containing results
        output_dir: directory to save plots
        training_methods: list of training methods to plot
        marker_alpha: alpha value for markers
        line_alpha: alpha value for lines
        linewidth: line width
        markersize: marker size
    """
    if training_methods is None:
        training_methods = ['energy', 'pruning', 'random_initialization', 'reoptimize_different_states']
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define J,h combinations
    J_h_combinations = [(0.4, 0.6), (0.45, 0.55)]
    
    # Create figure with two subplots sharing y-axis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    axes = [ax1, ax2]
    
    print("Creating two-panel energy density above ground state vs system size plot")
    
    # First pass: collect all data to determine y-axis limits
    all_energy_densities = []
    
    for J, h in J_h_combinations:
        # Define system sizes to check (from cluster_job_generator_data_collection.py)
        system_sizes = [(4, 2), (8, 4), (12, 6), (16, 8), (20, 10), (24, 12), (28, 14)]
        single_qubit_noise = 0.0
        two_qubit_noise = 0.0
        
        # Collect data for each training method and system size
        for training_method in training_methods:
            for system_qubits, bath_qubits in system_sizes:
                filepath = find_variational_cooling_file(results_dir, system_qubits, bath_qubits, J, h, 
                                                       MAX_NUM_SWEEPS, single_qubit_noise, two_qubit_noise, training_method)
                
                if filepath:
                    try:
                        # Load data
                        with open(filepath, 'r') as f:
                            data = json.load(f)
                        
                        # Get measurements from final sweep
                        final_sweep_key = f"sweep_{NUM_SWEEPS_STEADY_STATE}"
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
                                all_energy_densities.append(energy_density_above_gs)
                                
                            except (json.JSONDecodeError, KeyError):
                                continue
                    except (json.JSONDecodeError, KeyError, FileNotFoundError):
                        continue
    
    # Plot each J,h combination
    for panel_idx, (J, h) in enumerate(J_h_combinations):
        ax = axes[panel_idx]
        
        # Call the existing function with the specific axis
        plot_energy_density_vs_system_size(results_dir, output_dir, J, h, training_methods,
                                          marker_alpha, line_alpha, linewidth, markersize, ax)
        
        # Update title for each panel
        ax.set_title(f'J={J}, h={h}', fontsize=16)
        
        # Only show y-label on the left panel
        if panel_idx == 0:
            ax.set_ylabel('Energy Density Above Ground State', fontsize=14)
        else:
            ax.set_ylabel('')  # Remove y-label from right panel
    
    # Set shared y-axis limits to start at 0 and cover all data
    if all_energy_densities:
        y_min = 0
        y_max = max(all_energy_densities) * 1.05  # Add 5% padding at the top
        ax1.set_ylim(y_min, y_max)
        ax2.set_ylim(y_min, y_max)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    filename = "energy_density_vs_system_size_two_panels"
    filepath_pdf = os.path.join(output_dir, filename+".pdf")
    filepath_png = os.path.join(output_dir, filename+".png")
    plt.savefig(filepath_pdf, dpi=300, bbox_inches='tight')
    plt.savefig(filepath_png, dpi=300, bbox_inches='tight')
    
    print(f"Two-panel plot saved to: {filepath_pdf}")
    print(f"Two-panel plot saved to: {filepath_png}")
    
    plt.show()


def create_alternative_analysis_plots(results_dir: str = "results", 
                                     output_dir: str = "plots/alternative_analysis",
                                     training_methods: List[str] = None,
                                     marker_alpha: float = 1., line_alpha: float = 1.) -> None:
    """
    Create all analysis plots for alternative training methods.
    
    Args:
        results_dir: directory containing results
        output_dir: directory to save plots
        training_methods: list of training methods to analyze
        marker_alpha: alpha value for markers
        line_alpha: alpha value for lines
    """
    if training_methods is None:
        training_methods = ['energy', 'pruning', 'random_initialization', 'reoptimize_different_states']
    
    print("Creating analysis plots for alternative training methods...")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Two-panel energy density above ground state vs system size plot
    print("\n1. Creating two-panel energy density above ground state vs system size plot...")
    plot_energy_density_vs_system_size_two_panels(results_dir, output_dir, training_methods,
                                                 marker_alpha=marker_alpha, line_alpha=line_alpha,
                                                 linewidth=2, markersize=10)
    
    # 2. Individual plots for each J,h combination
    J_h_combinations = [(0.4, 0.6), (0.45, 0.55)]
    
    for J, h in J_h_combinations:
        print(f"\n2. Creating individual plot for J={J}, h={h}...")
        plot_energy_density_vs_system_size(results_dir, output_dir, J, h, training_methods,
                                          marker_alpha=marker_alpha, line_alpha=line_alpha,
                                          linewidth=2, markersize=10)
    
    # 3. Energy density vs sweeps plots for each J,h combination (using largest system size)
    print("\n3. Creating energy density vs sweeps plots...")
    largest_system_size = (4, 2)  # Use the largest system size from the alternative track
    
    for J, h in J_h_combinations:
        print(f"\n3. Creating energy density vs sweeps plot for J={J}, h={h}...")
        plot_energy_density_vs_sweeps(results_dir, output_dir, J, h, 
                                     system_qubits=largest_system_size[0], 
                                     bath_qubits=largest_system_size[1],
                                     training_methods=training_methods,
                                     marker_alpha=marker_alpha, line_alpha=line_alpha,
                                     linewidth=2, markersize=10)
    
    print("\n" + "=" * 60)
    print("All alternative training analysis plots created successfully!")
    print(f"Plots saved to: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze alternative training methods results')
    parser.add_argument('--results-dir', default='results', help='Directory containing results')
    parser.add_argument('--output-dir', default='plots/alternative_analysis', help='Output directory for plots')
    parser.add_argument('--training-methods', nargs='+', 
                       default=['energy', 'pruning', 'random_initialization', 'reoptimize_different_states'],
                       help='Training methods to analyze')
    parser.add_argument('--marker-alpha', type=float, default=1.0, help='Alpha value for markers')
    parser.add_argument('--line-alpha', type=float, default=1.0, help='Alpha value for lines')
    
    args = parser.parse_args()
    
    print("Alternative Training Methods Analysis")
    print("=" * 50)
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Training methods: {args.training_methods}")
    print()
    
    # Create all analysis plots
    create_alternative_analysis_plots(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        training_methods=args.training_methods,
        marker_alpha=args.marker_alpha,
        line_alpha=args.line_alpha
    )
