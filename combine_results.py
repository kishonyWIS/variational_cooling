#!/usr/bin/env python3
"""
Script to combine results from multiple variational cooling job CSV files.
"""

import pandas as pd
import os
import glob
import argparse


def combine_job_results(results_dir="results", output_file="combined_results.csv"):
    """
    Combine all job result CSV files into a single file.
    
    Args:
        results_dir: directory containing job result files
        output_file: output file for combined results
    """
    
    # Find all CSV files in results directory
    csv_pattern = os.path.join(results_dir, "results_job_*.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print(f"No job result files found in {results_dir}")
        print(f"Expected pattern: {csv_pattern}")
        return
    
    print(f"Found {len(csv_files)} job result files:")
    for f in sorted(csv_files):
        print(f"  {f}")
    
    # Read and combine all CSV files
    all_data = []
    for csv_file in sorted(csv_files):
        try:
            df = pd.read_csv(csv_file)
            # Add job ID column based on filename
            job_id = int(csv_file.split('_')[-1].replace('.csv', ''))
            df['job_id'] = job_id
            all_data.append(df)
            print(f"  Loaded {len(df)} rows from {csv_file}")
        except Exception as e:
            print(f"  Error loading {csv_file}: {e}")
    
    if not all_data:
        print("No valid data found in any files")
        return
    
    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Sort by job_id and other relevant columns
    combined_df = combined_df.sort_values(['job_id', 'system_qubits', 'bath_qubits', 'num_sweeps'])
    
    # Save combined results
    combined_df.to_csv(output_file, index=False)
    
    print(f"\nCombined results saved to: {output_file}")
    print(f"Total rows: {len(combined_df)}")
    print(f"Unique job IDs: {combined_df['job_id'].nunique()}")
    
    # Print summary statistics
    print("\nSummary by job ID:")
    job_summary = combined_df.groupby('job_id').agg({
        'system_qubits': 'count',
        'converged': lambda x: (x == True).sum() if 'converged' in x.name else 0,
        'total_time_minutes': 'sum'
    }).rename(columns={'system_qubits': 'parameter_sets', 'converged': 'converged_sets'})
    
    print(job_summary)
    
    # Print overall statistics
    print(f"\nOverall statistics:")
    print(f"  Total parameter sets: {len(combined_df)}")
    if 'converged' in combined_df.columns:
        converged_count = (combined_df['converged'] == True).sum()
        print(f"  Converged sets: {converged_count}")
        print(f"  Convergence rate: {converged_count/len(combined_df)*100:.1f}%")
    
    if 'total_time_minutes' in combined_df.columns:
        total_time = combined_df['total_time_minutes'].sum()
        print(f"  Total runtime: {total_time:.1f} minutes ({total_time/60:.1f} hours)")
    
    return combined_df


def analyze_results(csv_file="combined_results.csv"):
    """
    Analyze the combined results and generate summary statistics.
    
    Args:
        csv_file: path to combined results CSV file
    """
    
    if not os.path.exists(csv_file):
        print(f"Results file not found: {csv_file}")
        return
    
    df = pd.read_csv(csv_file)
    print(f"Analyzing {len(df)} parameter sets from {csv_file}")
    
    # Summary by system size
    print("\n=== Summary by System Size ===")
    size_summary = df.groupby(['system_qubits', 'bath_qubits']).agg({
        'converged': lambda x: (x == True).sum() if 'converged' in x.name else 0,
        'total_time_minutes': 'mean',
        'final_bond_dim': 'mean',
        'final_combined_std': 'mean'
    }).round(3)
    print(size_summary)
    
    # Summary by number of sweeps
    print("\n=== Summary by Number of Sweeps ===")
    sweep_summary = df.groupby('num_sweeps').agg({
        'converged': lambda x: (x == True).sum() if 'converged' in x.name else 0,
        'total_time_minutes': 'mean',
        'final_bond_dim': 'mean',
        'final_combined_std': 'mean'
    }).round(3)
    print(sweep_summary)
    
    # Summary by noise level
    print("\n=== Summary by Noise Level ===")
    noise_summary = df.groupby(['single_qubit_gate_noise', 'two_qubit_gate_noise']).agg({
        'converged': lambda x: (x == True).sum() if 'converged' in x.name else 0,
        'total_time_minutes': 'mean',
        'final_bond_dim': 'mean',
        'final_combined_std': 'mean'
    }).round(3)
    print(noise_summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Combine and analyze variational cooling results')
    parser.add_argument('--results-dir', type=str, default='results', 
                       help='Directory containing job result files')
    parser.add_argument('--output-file', type=str, default='combined_results.csv',
                       help='Output file for combined results')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze the combined results')
    
    args = parser.parse_args()
    
    # Combine results
    combine_job_results(args.results_dir, args.output_file)
    
    # Analyze if requested
    if args.analyze:
        analyze_results(args.output_file) 