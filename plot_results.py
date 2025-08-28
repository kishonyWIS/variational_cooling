#!/usr/bin/env python3
"""
Energy density plotting script for variational cooling results.
Creates energy density vs sweeps plots with error bars for each system size and noise rate,
separated by (J, h) parameter combinations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import argparse
import os


def load_and_clean_data(csv_file):
    """Load and clean the results data."""
    if not os.path.exists(csv_file):
        print(f"Results file not found: {csv_file}")
        return None
    
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} results from {csv_file}")
    
    # Convert numeric columns that might be strings
    df['combined_std'] = pd.to_numeric(df['combined_std'], errors='coerce')
    df['energy'] = pd.to_numeric(df['energy'], errors='coerce')
    df['ground_state_energy'] = pd.to_numeric(df['ground_state_energy'], errors='coerce')
    df['bond_dim'] = pd.to_numeric(df['bond_dim'], errors='coerce')
    
    # Filter to only include bond dimension 64
    df = df[df['bond_dim'] == 64]
    print(f"After filtering to bond dimension 64: {len(df)} results")
    
    # Remove rows with NaN values in critical columns
    df = df.dropna(subset=['combined_std', 'energy', 'ground_state_energy', 'bond_dim'])
    
    # Calculate derived quantities
    df['total_qubits'] = df['system_qubits'] + df['bath_qubits']
    df['noise_factor'] = df['single_qubit_gate_noise'] / 0.001  # Normalize to base noise
    
    return df


def plot_energy_density_vs_sweeps(df, J, h, output_dir="plots"):
    """Plot energy density vs number of sweeps with error bars for each system size, with noise levels as different colored lines."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter data for this J, h combination
    subset = df[(df['J'] == J) & (df['h'] == h)]
    
    if len(subset) == 0:
        print(f"No data found for J={J}, h={h}")
        return
    
    print(f"Plotting energy density vs sweeps for J={J}, h={h} with {len(subset)} data points")
    
    # Calculate energy density above ground state (energy per system qubit)
    subset['ground_state_energy_density'] = subset['ground_state_energy'] / subset['system_qubits']
    subset['energy_density_above_ground'] = (subset['energy'] - subset['ground_state_energy']) / subset['system_qubits']
    subset['energy_density_error'] = subset['combined_std'] / subset['system_qubits']
    
    # Get unique system sizes and noise factors
    system_sizes = sorted(subset['system_qubits'].unique())
    noise_factors = sorted(subset['noise_factor'].unique())
    
    # Create subplots: one for each system size
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Color map for noise levels
    color_array = plt.cm.get_cmap('viridis')(np.linspace(0, 1, len(noise_factors)))
    
    # Calculate global axis limits for consistency across all panels
    all_energy_densities = subset['energy_density_above_ground']
    all_sweeps = subset['num_sweeps']
    
    y_min, y_max = all_energy_densities.min(), all_energy_densities.max()
    y_range = y_max - y_min
    y_min_global = y_min - 0.1*y_range
    y_max_global = y_max + 0.1*y_range
    
    x_min, x_max = all_sweeps.min(), all_sweeps.max()
    x_range = x_max - x_min
    x_min_global = x_min - 0.05*x_range
    x_max_global = x_max + 0.05*x_range
    
    for i, system_size in enumerate(system_sizes):
        if i < len(axes):
            ax = axes[i]
            
            # Filter data for this system size
            system_subset = subset[subset['system_qubits'] == system_size]
            
            if len(system_subset) > 0:
                # Plot each noise level with a different color
                for j, noise_factor in enumerate(noise_factors):
                    noise_subset = system_subset[system_subset['noise_factor'] == noise_factor]
                    
                    if len(noise_subset) > 0:
                        # Sort by number of sweeps
                        sweep_data = noise_subset[['num_sweeps', 'energy_density_above_ground', 'energy_density_error']].copy()
                        sweep_data = sweep_data.sort_values('num_sweeps')
                        
                        # Plot with error bars
                        ax.errorbar(sweep_data['num_sweeps'], sweep_data['energy_density_above_ground'], 
                                  yerr=sweep_data['energy_density_error'], marker='o', capsize=3, capthick=1, 
                                  linewidth=2, markersize=6, color=color_array[j], alpha=0.8)
                
                # Add horizontal line at y=0 (ground state)
                ax.axhline(y=0, color='black', linestyle=':', alpha=0.5, label='Ground state')
                
                ax.set_xlabel('Number of Sweeps')
                ax.set_ylabel('Energy Density Above Ground State (ΔE/N)')
                ax.set_title(f'Energy Density vs Sweeps: {system_size} System Qubits')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                # Set consistent axis limits across all panels
                ax.set_xlim(x_min_global, x_max_global)
                ax.set_ylim(y_min_global, y_max_global)
    
    # Hide unused subplots
    for i in range(len(system_sizes), len(axes)):
        axes[i].set_visible(False)
    
    # Add a single colorbar for all subplots
    norm = colors.Normalize(min(noise_factors), max(noise_factors))
    sm = cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    
    # Add colorbar to the right of the entire figure
    cbar = plt.colorbar(sm, ax=axes, shrink=0.8, aspect=20, pad=0.25)
    cbar.set_label('Noise Factor', rotation=270, labelpad=15)
    cbar.ax.set_title('Noise Level', pad=10)
    
    # Add overall title with J, h parameters
    fig.suptitle(f'Energy Density vs Sweeps (J={J}, h={h})', fontsize=16, y=0.98)
    
    # Adjust layout to make room for colorbar
    plt.subplots_adjust(right=0.75, top=0.92)
    
    # Save with J, h in filename
    filename = f'energy_density_vs_sweeps_J{J}_h{h}.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.show()


def plot_steady_state_vs_noise(df, J, h, output_dir="plots"):
    """Plot steady state (asymptotic) energy density as a function of noise factor for different system sizes."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter data for this J, h combination
    subset = df[(df['J'] == J) & (df['h'] == h)]
    
    if len(subset) == 0:
        print(f"No data found for J={J}, h={h}")
        return
    
    print(f"Plotting steady state vs noise for J={J}, h={h} with {len(subset)} data points")
    
    # Calculate energy density above ground state (energy per system qubit)
    subset['ground_state_energy_density'] = subset['ground_state_energy'] / subset['system_qubits']
    subset['energy_density_above_ground'] = (subset['energy'] - subset['ground_state_energy']) / subset['system_qubits']
    subset['energy_density_error'] = subset['combined_std'] / subset['system_qubits']
    
    # Get unique system sizes and noise factors
    system_sizes = sorted(subset['system_qubits'].unique())
    noise_factors = sorted(subset['noise_factor'].unique())
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Color map for system sizes
    color_array = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(system_sizes)))
    
    # For each system size, find the steady state energy density at each noise level
    for i, system_size in enumerate(system_sizes):
        # Filter data for this system size
        system_subset = subset[subset['system_qubits'] == system_size]
        
        if len(system_subset) > 0:
            steady_state_energies = []
            steady_state_errors = []
            noise_levels = []
            
            # For each noise factor, find the energy density at the maximum number of sweeps (steady state)
            for noise_factor in noise_factors:
                noise_subset = system_subset[system_subset['noise_factor'] == noise_factor]
                
                if len(noise_subset) > 0:
                    # Find the maximum number of sweeps for this noise level
                    max_sweeps = noise_subset['num_sweeps'].max()
                    
                    # Get the data point with maximum sweeps (steady state)
                    steady_state_data = noise_subset[noise_subset['num_sweeps'] == max_sweeps]
                    
                    if len(steady_state_data) > 0:
                        # Take the mean if there are multiple points with same max sweeps
                        steady_state_energies.append(steady_state_data['energy_density_above_ground'].mean())
                        steady_state_errors.append(steady_state_data['energy_density_error'].mean())
                        noise_levels.append(noise_factor)
            
            if len(steady_state_energies) > 0:
                # Plot with error bars
                plt.errorbar(noise_levels, steady_state_energies, yerr=steady_state_errors, 
                           marker='o', capsize=3, capthick=1, linewidth=2, markersize=8, 
                           color=color_array[i], alpha=0.8, label=f'{system_size} qubits')
    
    # Add horizontal line at y=0 (ground state)
    plt.axhline(y=0, color='black', linestyle=':', alpha=0.5, label='Ground state')
    
    plt.xlabel('Noise Factor')
    plt.ylabel('Steady State Energy Density Above Ground State (ΔE/N)')
    plt.title(f'Steady State Energy Density vs Noise Factor (J={J}, h={h})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    # Save with J, h in filename
    filename = f'steady_state_vs_noise_J{J}_h{h}.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.show()


def plot_dual_points_vs_system_size(df, dual_pairs, output_dir="plots"):
    """
    Plot steady state energy density vs system size for dual points in the phase diagram.
    
    Args:
        df: DataFrame with results data
        dual_pairs: List of tuples, each containing two (J, h) pairs that are dual to each other
        output_dir: Output directory for plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate energy density above ground state for all data
    df['ground_state_energy_density'] = df['ground_state_energy'] / df['system_qubits']
    df['energy_density_above_ground'] = (df['energy'] - df['ground_state_energy']) / df['system_qubits']
    df['energy_density_error'] = df['combined_std'] / df['system_qubits']
    
    # Get unique noise factors for color mapping
    noise_factors = sorted(df['noise_factor'].unique())
    
    # Create color map for noise levels
    color_array = plt.cm.get_cmap('viridis')(np.linspace(0, 1, len(noise_factors)))
    
    for pair_idx, (point1, point2) in enumerate(dual_pairs):
        J1, h1 = point1
        J2, h2 = point2
        
        print(f"Plotting dual points: (J={J1}, h={h1}) and (J={J2}, h={h2})")
        
        # Filter data for both points
        subset1 = df[(df['J'] == J1) & (df['h'] == h1)]
        subset2 = df[(df['J'] == J2) & (df['h'] == h2)]
        
        if len(subset1) == 0 or len(subset2) == 0:
            print(f"Missing data for one or both points: (J={J1}, h={h1}) or (J={J2}, h={h2})")
            continue
        
        # Get unique system sizes
        all_system_sizes = sorted(set(subset1['system_qubits'].unique()) | set(subset2['system_qubits'].unique()))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot each noise factor with different colors
        for noise_idx, noise_factor in enumerate(noise_factors):
            color = color_array[noise_idx]
            
            # Plot point 1 (paramagnetic if J1 < h1, ferromagnetic if J1 > h1)
            phase1 = "paramagnetic" if J1 < h1 else "ferromagnetic"
            marker1 = 's' if J1 < h1 else '^'  # square for paramagnetic, triangle for ferromagnetic
            
            steady_state_energies1 = []
            steady_state_errors1 = []
            system_sizes1 = []
            
            for system_size in all_system_sizes:
                system_subset = subset1[(subset1['system_qubits'] == system_size) & 
                                      (subset1['noise_factor'] == noise_factor)]
                
                if len(system_subset) > 0:
                    # Find the maximum number of sweeps for this system size and noise level
                    max_sweeps = system_subset['num_sweeps'].max()
                    
                    # Get the data point with maximum sweeps (steady state)
                    steady_state_data = system_subset[system_subset['num_sweeps'] == max_sweeps]
                    
                    if len(steady_state_data) > 0:
                        # Take the mean if there are multiple points with same max sweeps
                        steady_state_energies1.append(steady_state_data['energy_density_above_ground'].mean())
                        steady_state_errors1.append(steady_state_data['energy_density_error'].mean())
                        system_sizes1.append(system_size)
            
            if len(steady_state_energies1) > 0:
                ax.errorbar(system_sizes1, steady_state_energies1, yerr=steady_state_errors1,
                           marker=marker1, capsize=3, capthick=1, linewidth=2, markersize=8,
                           color=color, alpha=0.8, 
                           label=f'(J={J1}, h={h1}) {phase1}, noise={noise_factor}')
            
            # Plot point 2 (paramagnetic if J2 < h2, ferromagnetic if J2 > h2)
            phase2 = "paramagnetic" if J2 < h2 else "ferromagnetic"
            marker2 = 's' if J2 < h2 else '^'  # square for paramagnetic, triangle for ferromagnetic
            
            steady_state_energies2 = []
            steady_state_errors2 = []
            system_sizes2 = []
            
            for system_size in all_system_sizes:
                system_subset = subset2[(subset2['system_qubits'] == system_size) & 
                                      (subset2['noise_factor'] == noise_factor)]
                
                if len(system_subset) > 0:
                    # Find the maximum number of sweeps for this system size and noise level
                    max_sweeps = system_subset['num_sweeps'].max()
                    
                    # Get the data point with maximum sweeps (steady state)
                    steady_state_data = system_subset[system_subset['num_sweeps'] == max_sweeps]
                    
                    if len(steady_state_data) > 0:
                        # Take the mean if there are multiple points with same max sweeps
                        steady_state_energies2.append(steady_state_data['energy_density_above_ground'].mean())
                        steady_state_errors2.append(steady_state_data['energy_density_error'].mean())
                        system_sizes2.append(system_size)
            
            if len(steady_state_energies2) > 0:
                ax.errorbar(system_sizes2, steady_state_energies2, yerr=steady_state_errors2,
                           marker=marker2, capsize=3, capthick=1, linewidth=2, markersize=8,
                           color=color, alpha=0.8, linestyle=':',  # dashed line for second point
                           label=f'(J={J2}, h={h2}) {phase2}, noise={noise_factor}')
        
        # Add horizontal line at y=0 (ground state)
        ax.axhline(y=0, color='black', linestyle=':', alpha=0.5, label='Ground state')
        
        ax.set_xlabel('System Size (Number of System Qubits)')
        ax.set_ylabel('Steady State Energy Density Above Ground State (ΔE/N)')
        ax.set_title(f'Steady State Energy Density vs System Size\nDual Points: (J={J1}, h={h1}) and (J={J2}, h={h2})')
        ax.grid(True, alpha=0.3)
        
        # Add legend with custom formatting
        handles, labels = ax.get_legend_handles_labels()
        # Group legend items by noise factor
        legend_elements = []
        # Add marker legend
        from matplotlib.lines import Line2D
        legend_elements.extend([
            Line2D([0], [0], marker='^', color='black', linestyle='-', markersize=8, label=f'Ferromagnetic (J = {J1}, h = {h1})'),
            Line2D([0], [0], marker='s', color='black', linestyle=':', markersize=8, label=f'Paramagnetic (J = {J2}, h = {h2})'),
            Line2D([0], [0], color='black', linestyle=':', label='Ground state')
        ])
        
        ax.legend(handles=legend_elements, loc='best', bbox_to_anchor=(1.05, 1))
        
        # Add colorbar for noise factors
        norm = colors.Normalize(min(noise_factors), max(noise_factors))
        sm = cm.ScalarMappable(cmap='viridis', norm=norm)
        sm.set_array([])
        # Add colorbar to the right of the entire figure, shift it slightly downwards
        cbar_ax = fig.add_axes([0.85, 0.07, 0.03, 0.7])  # [left, bottom, width, height]
        cbar = fig.colorbar(sm, cax=cbar_ax)
        # cbar = plt.colorbar(sm, ax=ax, shrink=0.8, aspect=20, pad=0.1)
        # pos = cbar.ax.get_position()
        # cbar.ax.set_position([pos.x0, pos.y0 - 0.3, pos.width, pos.height])
        cbar.set_label('Noise Factor', rotation=270, labelpad=15)
        cbar.ax.set_title('Noise Level', pad=10)

        plt.tight_layout()
        
        # Save with dual points in filename
        filename = f'dual_points_vs_system_size_pair{pair_idx+1}_J{J1}_h{h1}_J{J2}_h{h2}.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.show()


def plot_steady_state_vs_error_rate_all_jh(df, output_dir="plots"):
    """
    Plot steady state energy density vs error rate for all J,h combinations using the same color scheme as plot_correlations.py.
    
    Args:
        df: DataFrame with results data
        output_dir: Output directory for plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate energy density above ground state for all data
    df['ground_state_energy_density'] = df['ground_state_energy'] / df['system_qubits']
    df['energy_density_above_ground'] = (df['energy'] - df['ground_state_energy']) / df['system_qubits']
    df['energy_density_error'] = df['combined_std'] / df['system_qubits']
    
    # Get unique J,h combinations and sort by J values (same as plot_correlations.py)
    jh_combinations = df[['J', 'h']].drop_duplicates().values
    jh_combinations = sorted(jh_combinations, key=lambda x: x[0])  # Sort by J values
    
    if len(jh_combinations) == 0:
        print("No J,h combinations found in data")
        return
    
    print(f"Plotting steady state vs error rate for {len(jh_combinations)} J,h combinations")
    
    # Create colorbar using the same scheme as plot_correlations.py
    j_values = [jh[0] for jh in jh_combinations]
    norm = plt.Normalize(min(j_values), max(j_values))
    cmap = plt.cm.rainbow  # Same colormap as plot_correlations.py
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Get unique system sizes and noise factors
    system_sizes = sorted(df['system_qubits'].unique())
    noise_factors = sorted(df['noise_factor'].unique())
    
    # For each J,h combination, plot steady state energy density vs error rate (largest system size only)
    legend_handles = []
    legend_labels = []
    
    for jh_idx, (J, h) in enumerate(jh_combinations):
        print(f"Processing J={J}, h={h}")
        
        # Filter data for this J,h combination
        subset = df[(df['J'] == J) & (df['h'] == h)]
        
        if len(subset) == 0:
            print(f"No data found for J={J}, h={h}")
            continue
        
        # Use only the largest system size
        largest_system_size = max(system_sizes)
        system_subset = subset[subset['system_qubits'] == largest_system_size]
        
        if len(system_subset) > 0:
            steady_state_energies = []
            steady_state_errors = []
            error_rates = []
            
            # For each noise factor, find the energy density at the maximum number of sweeps (steady state)
            for noise_factor in noise_factors:
                noise_subset = system_subset[system_subset['noise_factor'] == noise_factor]
                
                if len(noise_subset) > 0:
                    # Find the maximum number of sweeps for this noise level
                    max_sweeps = noise_subset['num_sweeps'].max()
                    
                    # Get the data point with maximum sweeps (steady state)
                    steady_state_data = noise_subset[noise_subset['num_sweeps'] == max_sweeps]
                    
                    if len(steady_state_data) > 0:
                        # Take the mean if there are multiple points with same max sweeps
                        steady_state_energies.append(steady_state_data['energy_density_above_ground'].mean())
                        steady_state_errors.append(steady_state_data['energy_density_error'].mean())
                        # Convert noise factor back to actual error rate
                        error_rate = noise_factor * 0.001  # Base noise was 0.001
                        error_rates.append(error_rate)
            
            if len(steady_state_energies) > 0:
                # Get color from colorbar (same as plot_correlations.py)
                color = cmap(norm(J))
                
                # Plot with error bars
                lines = plt.errorbar(error_rates, steady_state_energies, yerr=steady_state_errors, 
                                   marker='o', capsize=3, capthick=1, linewidth=2, markersize=8, 
                                   color=color, alpha=0.8, label=f'J={J}, h={h}')
                
                # Store for legend (get the line object from the errorbar tuple)
                legend_handles.append(lines[0])
                legend_labels.append(f'J={J}, h={h}')
    
    # Add horizontal line at y=0 (ground state)
    plt.axhline(y=0, color='black', linestyle=':', alpha=0.5, label='Ground state')
    
    plt.xlabel('Noise Factor', fontsize=16)
    plt.ylabel('Steady State Energy Density Above Ground State (ΔE/N)', fontsize=16)
    plt.grid(True, alpha=0.3)
    
    # Add legend (same as plot_correlations.py)
    plt.legend(handles=legend_handles, labels=legend_labels, loc='best', fontsize=16)
    
    # Set font sizes for ticks
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    plt.tight_layout()
    
    # Save plot
    filename = 'steady_state_vs_error_rate_all_jh.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved as {os.path.join(output_dir, filename)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot energy density vs sweeps for variational cooling results')
    parser.add_argument('--csv-file', type=str, default='results/variational_cooling_results.csv',
                       help='Path to results CSV file')
    parser.add_argument('--output-dir', type=str, default='plots',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Load data
    df = load_and_clean_data(args.csv_file)
    if df is None:
        exit(1)
    
    # Get unique J, h combinations
    J_h_combinations = df[['J', 'h']].drop_duplicates().values
    print(f"Found {len(J_h_combinations)} unique (J, h) combinations:")
    for J, h in J_h_combinations:
        print(f"  J={J}, h={h}")
    
    # Generate plots for each J, h combination
    for J, h in J_h_combinations:
        print(f"\n=== Processing J={J}, h={h} ===")
        
        # Generate energy density plots
        print("Generating energy density vs sweeps plots...")
        plot_energy_density_vs_sweeps(df, J, h, args.output_dir)
        
        # Generate steady state vs noise plots
        print("Generating steady state energy density vs noise factor plots...")
        plot_steady_state_vs_noise(df, J, h, args.output_dir)
    
    # Generate dual points plots
    print("\n=== Generating dual points plots ===")
    
    # Define dual pairs: (J=0.6, h=0.4) and (J=0.4, h=0.6), and other pairs
    # First pair: paramagnetic (J=0.6, h=0.4) vs ferromagnetic (J=0.4, h=0.6)
    # Second pair: paramagnetic (J=0.55, h=0.45) vs ferromagnetic (J=0.45, h=0.55)
    dual_pairs = [
        ((0.6, 0.4), (0.4, 0.6)),  # First dual pair: paramagnetic vs ferromagnetic
        ((0.55, 0.45), (0.45, 0.55)),  # Second dual pair: paramagnetic vs ferromagnetic
    ]
    
    # Check if we have data for the dual pairs
    available_pairs = []
    for point1, point2 in dual_pairs:
        J1, h1 = point1
        J2, h2 = point2
        subset1 = df[(df['J'] == J1) & (df['h'] == h1)]
        subset2 = df[(df['J'] == J2) & (df['h'] == h2)]
        
        if len(subset1) > 0 and len(subset2) > 0:
            available_pairs.append((point1, point2))
            print(f"Found data for dual pair: (J={J1}, h={h1}) and (J={J2}, h={h2})")
        else:
            print(f"Missing data for dual pair: (J={J1}, h={h1}) and (J={J2}, h={h2})")
    
    if available_pairs:
        print("Generating dual points vs system size plots...")
        plot_dual_points_vs_system_size(df, available_pairs, args.output_dir)
    else:
        print("No dual pairs with available data found.")
    
    # Generate the new combined plot
    print("\n=== Generating steady state vs error rate plot for all J,h combinations ===")
    plot_steady_state_vs_error_rate_all_jh(df, args.output_dir)
    
    print(f"\nAll plots saved to {args.output_dir}/")
    print("Generated files:")
    for J, h in J_h_combinations:
        print(f"  energy_density_vs_sweeps_J{J}_h{h}.png")
        print(f"  steady_state_vs_noise_J{J}_h{h}.png")
    
    if available_pairs:
        for i, (point1, point2) in enumerate(available_pairs):
            J1, h1 = point1
            J2, h2 = point2
            print(f"  dual_points_vs_system_size_pair{i+1}_J{J1}_h{h1}_J{J2}_h{h2}.png")
    
    print("  steady_state_vs_error_rate_all_jh.png") 