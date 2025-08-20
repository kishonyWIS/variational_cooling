#!/usr/bin/env python3
import pandas as pd
import numpy as np

def analyze_existing_results():
    """Analyze existing results to show what's completed and what's missing."""
    
    # Load existing results
    df = pd.read_csv('results/variational_cooling_results.csv')
    
    print("=" * 80)
    print("VARIATIONAL COOLING PARAMETER COMBINATIONS STATUS")
    print("=" * 80)
    
    # Total results
    print(f"\nTotal existing results: {len(df)}")
    
    # Analysis by J,h values
    print("\nResults by J,h values:")
    jh_counts = df.groupby(['J', 'h']).size()
    for (J, h), count in jh_counts.items():
        print(f"  J={J}, h={h}: {count} results")
    
    # Analysis by system size
    print("\nResults by system size:")
    size_counts = df.groupby(['system_qubits', 'bath_qubits']).size()
    for (sys_qubits, bath_qubits), count in size_counts.items():
        print(f"  {sys_qubits}+{bath_qubits} qubits: {count} results")
    
    # Analysis by sweep count
    print("\nResults by sweep count:")
    sweep_counts = df.groupby('num_sweeps').size().sort_index()
    for sweep, count in sweep_counts.items():
        print(f"  {sweep} sweeps: {count} results")
    
    # Analysis by noise level
    print("\nResults by noise level (single_qubit_gate_noise):")
    noise_counts = df.groupby('single_qubit_gate_noise').size().sort_index()
    for noise, count in noise_counts.items():
        print(f"  {noise:.4f}: {count} results")
    
    # Calculate target combinations
    system_sizes = [(4, 2), (8, 4), (14, 7), (28, 14)]
    num_sweeps_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    J_h_list = [(0.6, 0.4), (0.55, 0.45), (0.45, 0.55), (0.4, 0.6)]
    noise_factors = np.linspace(0, 1, 11)
    
    total_target = len(system_sizes) * len(num_sweeps_list) * len(J_h_list) * len(noise_factors)
    
    print(f"\nTarget parameter space:")
    print(f"  System sizes: {len(system_sizes)} ({system_sizes})")
    print(f"  Sweep counts: {len(num_sweeps_list)} ({num_sweeps_list})")
    print(f"  J,h values: {len(J_h_list)} ({J_h_list})")
    print(f"  Noise factors: {len(noise_factors)} ({noise_factors})")
    print(f"  Total target combinations: {total_target}")
    
    # Calculate completion percentage
    completion_pct = (len(df) / total_target) * 100
    print(f"\nCompletion status:")
    print(f"  Completed: {len(df)} / {total_target} ({completion_pct:.1f}%)")
    print(f"  Missing: {total_target - len(df)} combinations")
    
    # Identify specific gaps
    print(f"\nSpecific gaps identified:")
    
    # Check for missing sweep counts
    existing_sweeps = set(df['num_sweeps'].unique())
    missing_sweeps = set(num_sweeps_list) - existing_sweeps
    if missing_sweeps:
        print(f"  Missing sweep counts: {sorted(missing_sweeps)}")
    else:
        print(f"  All sweep counts (0-12) are present")
    
    # Check for missing J,h combinations
    existing_jh = set(zip(df['J'], df['h']))
    missing_jh = set(J_h_list) - existing_jh
    if missing_jh:
        print(f"  Missing J,h combinations: {missing_jh}")
    else:
        print(f"  All J,h combinations are present")
    
    # Check for missing system sizes
    existing_sizes = set(zip(df['system_qubits'], df['bath_qubits']))
    missing_sizes = set(system_sizes) - existing_sizes
    if missing_sizes:
        print(f"  Missing system sizes: {missing_sizes}")
    else:
        print(f"  All system sizes are present")
    
    # Check for missing noise levels
    existing_noise = set(df['single_qubit_gate_noise'].round(6))
    target_noise = set(np.round(noise_factors * 0.001, 6))
    missing_noise = target_noise - existing_noise
    if missing_noise:
        print(f"  Missing noise levels: {sorted(missing_noise)}")
    else:
        print(f"  All noise levels are present")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    analyze_existing_results() 