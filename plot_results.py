#!/usr/bin/env python3
"""
Energy density plotting script for variational cooling results.
Creates energy density vs sweeps plots with error bars for each system size and noise rate.
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


def plot_energy_density_vs_sweeps(df, output_dir="plots"):
    """Plot energy density vs number of sweeps with error bars for each system size, with noise levels as different colored lines."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate energy density above ground state (energy per system qubit)
    df['ground_state_energy_density'] = df['ground_state_energy'] / df['system_qubits']
    df['energy_density_above_ground'] = (df['energy'] - df['ground_state_energy']) / df['system_qubits']
    df['energy_density_error'] = df['combined_std'] / df['system_qubits']
    
    # Get unique system sizes and noise factors
    system_sizes = sorted(df['system_qubits'].unique())
    noise_factors = sorted(df['noise_factor'].unique())
    
    # Create subplots: one for each system size
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Color map for noise levels
    color_array = plt.cm.get_cmap('viridis')(np.linspace(0, 1, len(noise_factors)))
    
    # Calculate global axis limits for consistency across all panels
    all_energy_densities = df['energy_density_above_ground']
    all_sweeps = df['num_sweeps']
    
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
            subset = df[df['system_qubits'] == system_size]
            
            if len(subset) > 0:
                # Plot each noise level with a different color
                for j, noise_factor in enumerate(noise_factors):
                    noise_subset = subset[subset['noise_factor'] == noise_factor]
                    
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
    
    # Adjust layout to make room for colorbar
    plt.subplots_adjust(right=0.75)
    plt.savefig(os.path.join(output_dir, 'energy_density_vs_sweeps.png'), dpi=300, bbox_inches='tight')
    plt.show()


def plot_steady_state_vs_noise(df, output_dir="plots"):
    """Plot steady state (asymptotic) energy density as a function of noise factor for different system sizes."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate energy density above ground state (energy per system qubit)
    df['ground_state_energy_density'] = df['ground_state_energy'] / df['system_qubits']
    df['energy_density_above_ground'] = (df['energy'] - df['ground_state_energy']) / df['system_qubits']
    df['energy_density_error'] = df['combined_std'] / df['system_qubits']
    
    # Get unique system sizes and noise factors
    system_sizes = sorted(df['system_qubits'].unique())
    noise_factors = sorted(df['noise_factor'].unique())
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Color map for system sizes
    color_array = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(system_sizes)))
    
    # For each system size, find the steady state energy density at each noise level
    for i, system_size in enumerate(system_sizes):
        # Filter data for this system size
        subset = df[df['system_qubits'] == system_size]
        
        if len(subset) > 0:
            steady_state_energies = []
            steady_state_errors = []
            noise_levels = []
            
            # For each noise factor, find the energy density at the maximum number of sweeps (steady state)
            for noise_factor in noise_factors:
                noise_subset = subset[subset['noise_factor'] == noise_factor]
                
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
    plt.title('Steady State Energy Density vs Noise Factor')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'steady_state_vs_noise.png'), dpi=300, bbox_inches='tight')
    plt.show()


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
    
    # Generate energy density plots
    print("Generating energy density vs sweeps plots...")
    plot_energy_density_vs_sweeps(df, args.output_dir)
    
    # Generate steady state vs noise plots
    print("Generating steady state energy density vs noise factor plots...")
    plot_steady_state_vs_noise(df, args.output_dir)
    
    print(f"Plots saved to {args.output_dir}/") 