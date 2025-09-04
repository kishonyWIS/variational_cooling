#!/usr/bin/env python3
"""
Cluster job generator for data_collection.py parameter sweep.
This script generates LSF job files to run each parameter set in a separate job.
"""

import json
import os
import sys
import numpy as np

# =============================================================================
# CONFIGURATION - All parameters defined here
# =============================================================================

# System sizes: [(4,2), (8,4), (12,6), (16,8), (20,10), (24,12), (28, 14)]
SYSTEM_SIZES = [(4, 2), (8, 4), (12, 6)]#, (16, 8), (20, 10), (24, 12), (28, 14)]

# J, h combinations: [(0.4,0.6), (0.45,0.55), (0.55, 0.45), (0.6, 0.4)]
J_H_LIST = [(0.4, 0.6), (0.45, 0.55), (0.55, 0.45), (0.6, 0.4)]

# Noise levels: (0.001, 0.01) * np.linspace(0, 1, 11)
NOISE_FACTORS = np.linspace(0, 1, 11)
BASE_SINGLE_QUBIT_NOISE = 0.001
BASE_TWO_QUBIT_NOISE = 0.01

# Fixed parameters
P = 3
NUM_SWEEPS = 12
TRAINING_METHOD = 'energy'
INITIAL_STATE = 'zeros'
BOND_DIMENSIONS = [32, 64]
BASE_SHOTS = 1000  # Base shots for BASE_TOTAL_SYSTEM_SIZE (28+14)
BASE_TOTAL_SYSTEM_SIZE = 28 + 14
OPEN_BOUNDARY = 1

# Job configuration
OUTPUT_DIR = "jobs"
JOB_NAME = "data_collection"
WALL_TIME = "24:00"  # Match cluster setting
MEMORY = "4GB"       # Increased for larger system sizes
CORES = 1
QUEUE = "berg"       # Change to berg queue for this cluster


def create_parameter_sets():
    """
    Create parameter sets for data_collection.py study.
    """
    parameter_sets = []
    
    # Generate all combinations
    for J, h in J_H_LIST:
        for noise_factor in NOISE_FACTORS:
            for system_qubits, bath_qubits in SYSTEM_SIZES:
                # Calculate shots based on system size
                current_total = system_qubits + bath_qubits
                
                # Calculate scaling factor (larger systems get fewer shots)
                scaling_factor = BASE_TOTAL_SYSTEM_SIZE / current_total
                num_shots = int(BASE_SHOTS * scaling_factor)
                
                parameter_sets.append({
                    'system_qubits': system_qubits,
                    'bath_qubits': bath_qubits,
                    'open_boundary': OPEN_BOUNDARY,
                    'J': J,
                    'h': h,
                    'p': P,
                    'num_sweeps': NUM_SWEEPS,
                    'single_qubit_gate_noise': BASE_SINGLE_QUBIT_NOISE * noise_factor,
                    'two_qubit_gate_noise': BASE_TWO_QUBIT_NOISE * noise_factor,
                    'training_method': TRAINING_METHOD,
                    'initial_state': INITIAL_STATE,
                    'bond_dimensions': BOND_DIMENSIONS,
                    'num_shots': num_shots
                })
    
    return parameter_sets


def create_job_script(job_id, params, output_dir="jobs", 
                     job_name="data_collection", wall_time="24:00", 
                     memory="4GB", cores=1, queue="berg"):
    """
    Create a job script for a single parameter set.
    
    Args:
        job_id: unique identifier for this job
        params: single parameter dictionary
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
        json.dump(params, f, indent=2)
    
    # Create the LSF job script
    with open(job_script, 'w') as f:
        f.write(f"""#!/bin/bash
#BSUB -J {job_name}_{job_id:03d}
#BSUB -q {queue}
#BSUB -W {wall_time}
#BSUB -R "rusage[mem=4096]"
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

# Run data collection for this parameter set
python3 data_collection.py --system-qubits {params['system_qubits']} \\
                          --bath-qubits {params['bath_qubits']} \\
                          --open-boundary {params['open_boundary']} \\
                          --J {params['J']} \\
                          --h {params['h']} \\
                          --p {params['p']} \\
                          --num-sweeps {params['num_sweeps']} \\
                          --single-qubit-noise {params['single_qubit_gate_noise']} \\
                          --two-qubit-noise {params['two_qubit_gate_noise']} \\
                          --training-method {params['training_method']} \\
                          --initial-state {params['initial_state']} \\
                          --bond-dimensions {','.join(map(str, params['bond_dimensions']))} \\
                          --num-shots {params['num_shots']} \\
                          --output-dir results

echo "Job {job_id} completed at $(date)"
""")
    
    return job_script, param_file


def generate_jobs(parameter_sets, output_dir="jobs", 
                 job_name="data_collection", wall_time="24:00", 
                 memory="4GB", cores=1, queue="berg"):
    """
    Generate job files, one for each parameter set.
    
    Args:
        parameter_sets: list of all parameter dictionaries
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
    
    # Create one job per parameter set
    for i, params in enumerate(parameter_sets):
        job_id = i + 1
        
        job_script, param_file = create_job_script(
            job_id, params, output_dir, job_name, wall_time, memory, cores, queue
        )
        job_scripts.append(job_script)
        
        print(f"Created job {job_id}: {job_script}")
        print(f"  Parameter file: {param_file}")
        print(f"  System: {params['system_qubits']}+{params['bath_qubits']} qubits")
        print(f"  J={params['J']}, h={params['h']}")
        print(f"  Noise: ({params['single_qubit_gate_noise']:.6f}, {params['two_qubit_gate_noise']:.6f})")
        print(f"  Shots: {params['num_shots']}")
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
        f.write("# Script to submit all data collection jobs\n\n")
        
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
        f.write("# Script to monitor data collection jobs\n\n")
        f.write("echo 'Job Status Summary:'\n")
        f.write("echo '=================='\n\n")
        
        f.write("# Check running jobs\n")
        f.write("echo 'Running jobs:'\n")
        f.write("bjobs | grep data_collection || echo 'No running jobs'\n\n")
        
        f.write("# Check pending jobs\n")
        f.write("echo 'Pending jobs:'\n")
        f.write("bjobs | grep PEND || echo 'No pending jobs'\n\n")
        
        f.write("# Check completed jobs\n")
        f.write("echo 'Completed jobs:'\n")
        f.write("ls -la results/*.json 2>/dev/null || echo 'No results files found'\n\n")
        
        f.write("# Count total parameter sets completed\n")
        f.write("echo 'Parameter sets completed:'\n")
        f.write("ls results/*.json 2>/dev/null | wc -l || echo 'No results files found'\n")
    
    # Make the script executable
    os.chmod(monitor_script, 0o755)
    
    return monitor_script


if __name__ == "__main__":
    print("Generating cluster jobs for data_collection.py parameter sweep...")
    print(f"Output directory: {OUTPUT_DIR}")
    print("Each parameter set will run in a separate job")
    
    # Check if jobs directory exists and warn user
    if os.path.exists(OUTPUT_DIR):
        print(f"Warning: Jobs directory '{OUTPUT_DIR}' already exists.")
        print("This will overwrite existing job files. Press Ctrl+C to cancel or Enter to continue...")
        try:
            input()
        except KeyboardInterrupt:
            print("Operation cancelled.")
            exit(0)
        print("Continuing with job generation...")
    
    print()
    
    # Get all parameter sets
    print("Generating all parameter sets...")
    all_parameter_sets = create_parameter_sets()
    print(f"Total parameter sets: {len(all_parameter_sets)}")
    
    # Print summary of parameter combinations
    print("\nParameter combinations summary:")
    print("=" * 50)
    
    # Summary by J,h values
    for J, h in J_H_LIST:
        count = sum(1 for params in all_parameter_sets if params['J'] == J and params['h'] == h)
        print(f"J={J}, h={h}: {count} combinations")
    
    # Summary by system size
    for sys_qubits, bath_qubits in SYSTEM_SIZES:
        count = sum(1 for params in all_parameter_sets 
                   if params['system_qubits'] == sys_qubits and params['bath_qubits'] == bath_qubits)
        print(f"{sys_qubits}+{bath_qubits} qubits: {count} combinations")
    
    # Summary by noise levels
    for noise_factor in NOISE_FACTORS:
        count = sum(1 for params in all_parameter_sets 
                   if abs(params['single_qubit_gate_noise'] - BASE_SINGLE_QUBIT_NOISE * noise_factor) < 1e-6)
        print(f"Noise factor {noise_factor:.1f}: {count} combinations")
    
    # Summary of shot allocation
    print("\nShot allocation by system size:")
    print("-" * 30)
    for sys_qubits, bath_qubits in SYSTEM_SIZES:
        current_total = sys_qubits + bath_qubits
        scaling_factor = BASE_TOTAL_SYSTEM_SIZE / current_total
        shots = int(BASE_SHOTS * scaling_factor)
        print(f"{sys_qubits}+{bath_qubits} qubits: {shots} shots (scaling factor: {scaling_factor:.2f})")
    
    print("=" * 50)
    
    # Generate job files
    print(f"\nGenerating job files for {len(all_parameter_sets)} parameter combinations...")
    job_scripts = generate_jobs(
        all_parameter_sets, 
        output_dir=OUTPUT_DIR,
        job_name=JOB_NAME,
        wall_time=WALL_TIME,
        memory=MEMORY,
        cores=CORES,
        queue=QUEUE
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
    
    print(f"\nTotal jobs created: {len(job_scripts)}")
    print(f"Total parameter combinations: {len(all_parameter_sets)}")
    print("Each parameter set runs in its own job")
