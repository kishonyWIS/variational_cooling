#!/usr/bin/env python3
"""
Plot correlations as a function of distance from CSV files with different J,h values.
Error bars combine bond dimension differences (32 vs 64) and shot noise.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from typing import Dict, Tuple, List

def load_csv_data(filepath: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load CSV data and separate bond dimension 32 and 64 results.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        Tuple of (data_32, data_64) DataFrames
    """
    df = pd.read_csv(filepath)
    
    # Separate data by bond dimension
    data_32 = df[df['bond_dim'] == 32].iloc[0] if len(df[df['bond_dim'] == 32]) > 0 else None
    data_64 = df[df['bond_dim'] == 64].iloc[0] if len(df[df['bond_dim'] == 64]) > 0 else None
    
    return data_32, data_64

def calculate_correlation_errors(data_32: pd.Series, data_64: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate correlation values and error bars.
    
    Error bars combine:
    1. Bond dimension difference (32 vs 64)
    2. Shot noise from correlation_i_std columns
    
    Args:
        data_32: Data for bond dimension 32
        data_64: Data for bond dimension 64
        
    Returns:
        Tuple of (correlation_values, error_bars)
    """
    if data_32 is None or data_64 is None:
        return None, None
    
    # Extract correlation values from bond dimension 64 (as requested)
    correlation_values = []
    shot_noise_errors = []
    
    # Find the number of correlations dynamically
    num_correlations = 0
    for i in range(100):  # Check up to 100 correlations
        corr_key = f'correlation_{i}_value'
        if corr_key in data_64:
            num_correlations = i + 1
        else:
            break
    
    if num_correlations == 0:
        return None, None, 0
    
    for i in range(num_correlations):
        corr_key = f'correlation_{i}_value'
        std_key = f'correlation_{i}_std'
        
        if corr_key in data_64 and std_key in data_64:
            corr_val = data_64[corr_key]
            shot_std = data_64[std_key]
            
            correlation_values.append(corr_val)
            shot_noise_errors.append(shot_std)
    
    correlation_values = np.array(correlation_values)
    shot_noise_errors = np.array(shot_noise_errors)
    
    # Calculate bond dimension difference errors
    bond_dim_errors = []
    for i in range(num_correlations):
        corr_key = f'correlation_{i}_value'
        
        if corr_key in data_32 and corr_key in data_64:
            # Difference between bond dim 32 and 64
            diff = abs(data_32[corr_key] - data_64[corr_key])
            bond_dim_errors.append(diff)
        else:
            bond_dim_errors.append(0.0)
    
    bond_dim_errors = np.array(bond_dim_errors)
    
    # Combine errors: sqrt(sum of squares)
    total_errors = np.sqrt(shot_noise_errors**2 + bond_dim_errors**2)
    
    return correlation_values, total_errors, num_correlations

def plot_correlations_vs_distance():
    """
    Main function to plot correlations vs distance for all J,h combinations.
    """
    # Find all CSV files with test_your_params pattern
    csv_files = glob.glob("test_your_params_J*.csv")
    
    if not csv_files:
        print("No CSV files found matching pattern 'test_your_params_J*.csv'")
        return
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    
    # Store J,h values and data for sorting
    jh_data = []
    
    for csv_file in sorted(csv_files):
        print(f"Processing {csv_file}...")
        
        # Extract J and h values from CSV content (more reliable)
        df = pd.read_csv(csv_file)
        j_val = df['J'].iloc[0]
        h_val = df['h'].iloc[0]
        
        # Load data
        data_32, data_64 = load_csv_data(csv_file)
        
        if data_32 is None or data_64 is None:
            print(f"Warning: Missing bond dimension data in {csv_file}")
            continue
        
        # Calculate correlations and errors
        correlations, errors, num_corrs = calculate_correlation_errors(data_32, data_64)
        
        if correlations is None:
            print(f"Warning: Could not calculate correlations for {csv_file}")
            continue
        
        # Store data for sorting
        jh_data.append({
            'j_val': j_val,
            'h_val': h_val,
            'correlations': correlations,
            'errors': errors,
            'num_corrs': num_corrs,
            'csv_file': csv_file
        })
    
    # Sort by increasing J values
    jh_data.sort(key=lambda x: x['j_val'])
    
    # Create colorbar
    j_values = [data['j_val'] for data in jh_data]
    norm = plt.Normalize(min(j_values), max(j_values))
    cmap = plt.cm.rainbow  # You can change this to other colormaps like 'plasma', 'inferno', etc.
    
    # Store legend information
    legend_handles = []
    legend_labels = []
    
    for i, data in enumerate(jh_data):
        j_val = data['j_val']
        h_val = data['h_val']
        correlations = data['correlations']
        errors = data['errors']
        num_corrs = data['num_corrs']
        
        # Distance values (1 to num_corrs) - correlation_0_value corresponds to distance 1
        distances = np.arange(1, num_corrs + 1)
        
        # Get color from colorbar
        color = cmap(norm(j_val))
        
        # Plot with error bars
        line, = plt.plot(distances, correlations, 'o-', color=color, 
                        linewidth=2, markersize=8, label=f'J={j_val}, h={h_val}')
        
        # Add error bars
        plt.errorbar(distances, correlations, yerr=errors, fmt='none', 
                    color=color, capsize=5, capthick=2, alpha=0.7)
        
        # Store for legend
        legend_handles.append(line)
        legend_labels.append(f'J={j_val}, h={h_val}')
    
    # Customize the plot
    plt.xlabel('|i-j|', fontsize=18)
    plt.ylabel('⟨σᵢᶻσⱼᶻ⟩', fontsize=18)
    plt.title('Correlations vs Distance for Different J,h Values', fontsize=20)
    plt.grid(True, alpha=0.3)
    plt.legend(handles=legend_handles, labels=legend_labels, loc='best', fontsize=16)
    
    # Set x-axis ticks to integer distances
    plt.xticks(distances, fontsize=16)
    
    # Set y-axis tick font size
    plt.yticks(fontsize=16)
    
    # Add horizontal line at y=0 for reference
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.8)
    # cbar.set_label('J value', fontsize=16)
    # cbar.ax.tick_params(labelsize=14)
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save the plot
    output_filename = 'correlations_vs_distance.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {output_filename}")
    
    # Show the plot
    plt.show()

def print_summary_statistics():
    """
    Print summary statistics for each J,h combination.
    """
    csv_files = glob.glob("test_your_params_J*.csv")
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    for csv_file in sorted(csv_files):
        print(f"\nFile: {csv_file}")
        
        data_32, data_64 = load_csv_data(csv_file)
        
        if data_32 is None or data_64 is None:
            print("  Missing bond dimension data")
            continue
        
        # Extract J and h values
        j_val = data_64['J']
        h_val = data_64['h']
        
        print(f"  J = {j_val}, h = {h_val}")
        print(f"  Bond dim 32 energy: {data_32['energy']:.6f}")
        print(f"  Bond dim 64 energy: {data_64['energy']:.6f}")
        print(f"  Energy difference: {abs(data_32['energy'] - data_64['energy']):.6f}")
        
        # Show correlation values
        print("  Correlations (bond dim 64):")
        # Find the number of correlations dynamically
        num_corrs = 0
        for i in range(100):  # Check up to 100 correlations
            corr_key = f'correlation_{i}_value'
            if corr_key in data_64:
                num_corrs = i + 1
            else:
                break
        
        for i in range(num_corrs):
            corr_key = f'correlation_{i}_value'
            std_key = f'correlation_{i}_std'
            if corr_key in data_64:
                print(f"    Distance {i+1}: {data_64[corr_key]:.6f} ± {data_64[std_key]:.6f}")

if __name__ == "__main__":
    print("Loading and plotting correlation data...")
    
    # Print summary statistics first
    print_summary_statistics()
    
    # Create the main plot
    plot_correlations_vs_distance()
    
    print("\nPlotting complete!")
