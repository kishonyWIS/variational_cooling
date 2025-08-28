#!/usr/bin/env python3
"""
Cluster job generator for variational cooling parameter sweep.
This script generates LSF job files to run different parameter sets in separate jobs.
"""

import json
import os
import sys
import pandas as pd
import numpy as np
from variational_cooling_mps_simulation import create_example_parameter_sets


def load_existing_results(csv_file):
    """Load existing results from CSV file."""
    if not os.path.exists(csv_file):
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(csv_file)
        return df
    except Exception as e:
        print(f"Warning: Could not read CSV file {csv_file}: {e}")
        return pd.DataFrame()


def filter_missing_combinations(parameter_sets, existing_df):
    """
    Filter parameter sets to only include combinations that are missing from existing results.
    
    Args:
        parameter_sets: list of parameter dictionaries
        existing_df: DataFrame of existing results
    
    Returns:
        list of parameter dictionaries for missing combinations
    """
    if existing_df.empty:
        print("No existing results found, running all parameter sets")
        return parameter_sets
    
    # Create a set of existing combinations for fast lookup
    existing_combinations = set()
    
    for _, row in existing_df.iterrows():
        try:
            # Skip error rows
            if (row.get('ground_state_energy') == 'ERROR' or 
                pd.isna(row.get('ground_state_energy')) or
                str(row.get('ground_state_energy')).startswith('ERROR')):
                continue
                
            # Create a tuple of key parameters for comparison
            # Round floating point values to handle precision issues
            combo = (
                int(float(row['system_qubits'])),
                int(float(row['bath_qubits'])),
                round(float(row['J']), 6),
                round(float(row['h']), 6),
                int(float(row['num_sweeps'])),
                round(float(row['single_qubit_gate_noise']), 6),
                round(float(row['two_qubit_gate_noise']), 6)
            )
            existing_combinations.add(combo)
        except (ValueError, KeyError, TypeError) as e:
            # Skip rows with conversion errors
            continue
    
    # Filter parameter sets to only include missing combinations
    missing_combinations = []
    
    for params in parameter_sets:
        combo = (
            params['system_qubits'],
            params['bath_qubits'],
            round(params['J'], 6),
            round(params['h'], 6),
            params['num_sweeps'],
            round(params['single_qubit_gate_noise'], 6),
            round(params['two_qubit_gate_noise'], 6)
        )
        
        if combo not in existing_combinations:
            missing_combinations.append(params)
    
    return missing_combinations


def create_job_script(job_id, parameter_sets, output_dir="jobs", 
                     job_name="variational_cooling", wall_time="24:00", 
                     memory="8GB", cores=1, queue="normal", bond_dims="32,64"):
    """
    Create a job script for a subset of parameter sets.
    
    Args:
        job_id: unique identifier for this job
        parameter_sets: list of parameter dictionaries to run
        output_dir: directory to save job files
        job_name: name for the job
        wall_time: wall clock time limit (HH:MM format)
        memory: memory requirement
        cores: number of cores
        queue: queue name
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create job script filename
    job_script = os.path.join(output_dir, f"job_{job_id:03d}.lsf")
    
    # Create parameter file
    param_file = os.path.join(output_dir, f"params_{job_id:03d}.json")
    with open(param_file, 'w') as f:
        json.dump(parameter_sets, f, indent=2)
    
    # Create the LSF job script
    with open(job_script, 'w') as f:
        f.write(f"""#!/bin/bash
#BSUB -J {job_name}_{job_id:03d}
#BSUB -q {queue}
#BSUB -W {wall_time}
#BSUB -M {memory}
#BSUB -n {cores}
#BSUB -o {output_dir}/job_{job_id:03d}.out
#BSUB -e {output_dir}/job_{job_id:03d}.err
#BSUB -R "span[hosts=1]"

# Load any necessary modules (modify as needed for your cluster)
# module load python/3.9
# module load qiskit

# Set up environment
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Create results directory
mkdir -p results

# Run the parameter sweep for this job
python3 variational_cooling_mps_simulation.py --job-id {job_id} --param-file {param_file} --output-dir results --shared-csv results/variational_cooling_results.csv --bond-dims {bond_dims} --include-correlations --analyze-correlations

echo "Job {job_id} completed at $(date)"
""")
    
    return job_script, param_file


def generate_jobs(parameter_sets, jobs_per_file=1, output_dir="jobs", 
                 job_name="variational_cooling", wall_time="24:00", 
                 memory="8GB", cores=1, queue="normal", bond_dims="32,64"):
    """
    Generate multiple job files, each containing a subset of parameter sets.
    
    Args:
        parameter_sets: list of all parameter dictionaries
        jobs_per_file: number of parameter sets to run per job
        output_dir: directory to save job files
        job_name: name for the jobs
        wall_time: wall clock time limit
        memory: memory requirement
        cores: number of cores
        queue: queue name
    
    Returns:
        list of job script filenames
    """
    
    job_scripts = []
    
    # Split parameter sets into chunks
    for i in range(0, len(parameter_sets), jobs_per_file):
        job_id = i // jobs_per_file + 1
        chunk = parameter_sets[i:i + jobs_per_file]
        
        job_script, param_file = create_job_script(
            job_id, chunk, output_dir, job_name, wall_time, memory, cores, queue, bond_dims
        )
        job_scripts.append(job_script)
        
        print(f"Created job {job_id}: {job_script}")
        print(f"  Parameter file: {param_file}")
        print(f"  Parameter sets: {len(chunk)}")
        print()
    
    return job_scripts


def create_submit_all_script(job_scripts, output_dir="jobs"):
    """
    Create a script to submit all jobs.
    
    Args:
        job_scripts: list of job script filenames
        output_dir: directory containing job files
    """
    
    submit_script = os.path.join(output_dir, "submit_all.sh")
    
    with open(submit_script, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Script to submit all variational cooling jobs\n\n")
        
        for job_script in job_scripts:
            f.write(f"bsub < {job_script}\n")
            f.write("sleep 1  # Small delay between submissions\n")
    
    # Make the script executable
    os.chmod(submit_script, 0o755)
    
    return submit_script


def create_monitor_script(job_scripts, output_dir="jobs"):
    """
    Create a script to monitor job status.
    
    Args:
        job_scripts: list of job script filenames
        output_dir: directory containing job files
    """
    
    monitor_script = os.path.join(output_dir, "monitor_jobs.sh")
    
    with open(monitor_script, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Script to monitor variational cooling jobs\n\n")
        f.write("echo 'Job Status Summary:'\n")
        f.write("echo '=================='\n\n")
        
        f.write("# Check running jobs\n")
        f.write("echo 'Running jobs:'\n")
        f.write("bjobs | grep variational_cooling || echo 'No running jobs'\n\n")
        
        f.write("# Check pending jobs\n")
        f.write("echo 'Pending jobs:'\n")
        f.write("bjobs | grep PEND || echo 'No pending jobs'\n\n")
        
        f.write("# Check completed jobs\n")
        f.write("echo 'Completed jobs:'\n")
        f.write("ls -la results/*.csv 2>/dev/null || echo 'No results files found'\n\n")
        
        f.write("# Count total parameter sets completed\n")
        f.write("echo 'Parameter sets completed:'\n")
        f.write("wc -l results/*.csv 2>/dev/null | tail -1 || echo 'No results files found'\n")
    
    # Make the script executable
    os.chmod(monitor_script, 0o755)
    
    return monitor_script


if __name__ == "__main__":
    # Configuration
    JOBS_PER_FILE = 8
    OUTPUT_DIR = "jobs"
    JOB_NAME = "variational_cooling"
    WALL_TIME = "24:00"
    MEMORY = "1GB"
    CORES = 1
    QUEUE = "berg"
    BOND_DIMS = "32,64"

    RESULTS_CSV = "results/variational_cooling_results.csv"
    
    print("Generating cluster jobs for variational cooling parameter sweep...")
    print(f"Jobs per file: {JOBS_PER_FILE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Results CSV: {RESULTS_CSV}")
    
    # Clean up existing job files
    if os.path.exists(OUTPUT_DIR):
        import shutil
        shutil.rmtree(OUTPUT_DIR)
        print(f"Cleaned up existing directory: {OUTPUT_DIR}")
    
    print()
    
    # Load existing results
    print("Loading existing results...")
    existing_df = load_existing_results(RESULTS_CSV)
    if not existing_df.empty:
        print(f"Found {len(existing_df)} existing results")
    else:
        print("No existing results found")
    
    # Get all parameter sets
    print("Generating all parameter sets...")
    all_parameter_sets = create_example_parameter_sets()
    print(f"Total parameter sets: {len(all_parameter_sets)}")
    
    # Filter to only missing combinations
    print("Filtering to missing combinations...")
    missing_parameter_sets = filter_missing_combinations(all_parameter_sets, existing_df)
    print(f"Missing parameter sets: {len(missing_parameter_sets)}")
    
    if len(missing_parameter_sets) == 0:
        print("No missing combinations found! All parameter sets are already completed.")
        sys.exit(0)
    
    # Print summary of missing combinations
    print("\nMissing combinations summary:")
    print("=" * 50)
    
    # Summary by J,h values
    J_h_list = [(0.6, 0.4), (0.55, 0.45), (0.45, 0.55), (0.4, 0.6)]
    for J, h in J_h_list:
        count = sum(1 for params in missing_parameter_sets if params['J'] == J and params['h'] == h)
        print(f"J={J}, h={h}: {count} missing combinations")
    
    # Summary by system size
    system_sizes = [(4, 2), (8, 4), (14, 7), (28, 14)]
    for sys_qubits, bath_qubits in system_sizes:
        count = sum(1 for params in missing_parameter_sets 
                   if params['system_qubits'] == sys_qubits and params['bath_qubits'] == bath_qubits)
        print(f"{sys_qubits}+{bath_qubits} qubits: {count} missing combinations")
    
    # Summary by sweep count
    sweep_counts = sorted(set(params['num_sweeps'] for params in missing_parameter_sets))
    for sweep in sweep_counts:
        count = sum(1 for params in missing_parameter_sets if params['num_sweeps'] == sweep)
        print(f"{sweep} sweeps: {count} missing combinations")
    
    print("=" * 50)
    
    # Generate job files
    print(f"\nGenerating job files for {len(missing_parameter_sets)} missing combinations...")
    job_scripts = generate_jobs(
        missing_parameter_sets, 
        jobs_per_file=JOBS_PER_FILE,
        output_dir=OUTPUT_DIR,
        job_name=JOB_NAME,
        wall_time=WALL_TIME,
        memory=MEMORY,
        cores=CORES,
        queue=QUEUE,
        bond_dims=BOND_DIMS
    )
    
    # Create submit all script
    submit_script = create_submit_all_script(job_scripts, OUTPUT_DIR)
    print(f"Submit script: {submit_script}")
    
    # Create monitor script
    monitor_script = create_monitor_script(job_scripts, OUTPUT_DIR)
    print(f"Monitor script: {monitor_script}")
    
    print(f"\nTo submit all jobs:")
    print(f"  {submit_script}")
    print(f"\nTo monitor jobs:")
    print(f"  {monitor_script}")
    print(f"\nTo submit individual jobs:")
    for job_script in job_scripts:
        print(f"  bsub < {job_script}") 