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

# Study configurations
STUDY_CONFIGS = {
    'original': {
        'system_sizes': [(4, 2), (8, 4), (12, 6), (16, 8), (20, 10), (24, 12), (28, 14)],
        'j_h_list': [(0.4, 0.6), (0.45, 0.55), (0.55, 0.45), (0.6, 0.4)],
        'noise_factors': np.linspace(0, 1, 11),
        'training_methods': ['energy'],
        'description': 'comprehensive parameter sweep'
    },
    'alternative': {
        'system_sizes': [(4, 2), (8, 4), (12, 6), (16, 8), (20, 10), (24, 12), (28, 14)],
        'j_h_list': [(0.4, 0.6), (0.45, 0.55)],
        'noise_factors': [0.0],
        'training_methods': ["pruning", "random_initialization", "reoptimize_different_states"],
        'description': 'alternative training methods study'
    }
}

# Common parameters
BASE_SINGLE_QUBIT_NOISE = 0.001
BASE_TWO_QUBIT_NOISE = 0.01
P = 3
NUM_SWEEPS = 40
INITIAL_STATE = 'zeros'
BOND_DIMENSIONS = [32, 64]
BASE_SHOTS_LARGE = 1000  # Base shots for BASE_TOTAL_SYSTEM_SIZE (28+14)
BASE_SHOTS_SMALL = 100  # Base shots for BASE_TOTAL_SYSTEM_SIZE (28+14)
BASE_TOTAL_SYSTEM_SIZE = 28 + 14
OPEN_BOUNDARY = 1

# Job configuration
WALL_TIME = "96:00"  # 4 days - extended to prevent timeouts
MEMORY = "4GB"       # Increased for larger system sizes
CORES = 1
QUEUE = "berg"       # Change to berg queue for this cluster


def create_parameter_sets(study_type="original"):
    """
    Create parameter sets for data_collection.py study.
    
    Args:
        study_type: "original" for comprehensive parameter sweep, 
                   "alternative" for alternative training methods study
    
    Returns:
        list: List of parameter dictionaries for each combination
    """
    if study_type not in STUDY_CONFIGS:
        raise ValueError(f"Unknown study_type: {study_type}. Available: {list(STUDY_CONFIGS.keys())}")
    
    config = STUDY_CONFIGS[study_type]
    parameter_sets = []
    
    # Generate all combinations
    for J, h in config['j_h_list']:
        for noise_factor in config['noise_factors']:
            for system_qubits, bath_qubits in config['system_sizes']:
                for training_method in config['training_methods']:
                    # Calculate shots based on system size
                    current_total = system_qubits + bath_qubits
                    
                    # Calculate scaling factor (larger systems get fewer shots)
                    scaling_factor = BASE_TOTAL_SYSTEM_SIZE / current_total
                    if current_total <= 12+6:
                        num_shots = int(BASE_SHOTS_LARGE * scaling_factor)
                    else:
                        num_shots = int(BASE_SHOTS_SMALL * scaling_factor)
                    
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
                        'training_method': training_method,
                        'initial_state': INITIAL_STATE,
                        'bond_dimensions': BOND_DIMENSIONS,
                        'num_shots': num_shots
                    })
    
    return parameter_sets


def create_job_script(job_id, params, output_dir="jobs", 
                     job_name="data_collection", wall_time="96:00", 
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
                 job_name="data_collection", wall_time="96:00", 
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


def normalize_float_for_filename(value, max_precision=6):
    """
    Normalize a float value for consistent filename representation.
    Handles floating-point precision issues by rounding to a reasonable precision
    and removing trailing zeros.
    
    Args:
        value: float value to normalize
        max_precision: maximum number of decimal places to consider
    
    Returns:
        str: normalized string representation
    """
    # Round to max_precision decimal places to handle floating-point errors
    rounded_value = round(value, max_precision)
    
    # Convert to string with max_precision decimal places
    str_value = f"{rounded_value:.{max_precision}f}"
    
    # Remove trailing zeros and decimal point if not needed
    str_value = str_value.rstrip('0').rstrip('.')
    
    return str_value


def generate_variational_cooling_filename(system_qubits, bath_qubits, J, h, num_sweeps, 
                                         single_qubit_gate_noise, two_qubit_gate_noise, 
                                         training_method):
    """
    Generate consistent filename for variational cooling data files.
    Handles floating-point precision issues by using proper normalization.
    
    Args:
        system_qubits: number of system qubits
        bath_qubits: number of bath qubits
        J: Ising coupling strength
        h: transverse field strength
        num_sweeps: number of sweeps
        single_qubit_gate_noise: single qubit gate noise level
        two_qubit_gate_noise: two qubit gate noise level
        training_method: training method name
    
    Returns:
        str: consistent filename
    """
    # Normalize noise values to avoid floating-point precision issues
    single_qubit_noise_str = normalize_float_for_filename(single_qubit_gate_noise)
    two_qubit_noise_str = normalize_float_for_filename(two_qubit_gate_noise)
    
    return f"variational_cooling_data_sys{system_qubits}_bath{bath_qubits}_J{J}_h{h}_sweeps{num_sweeps}_noise{single_qubit_noise_str}_{two_qubit_noise_str}_method{training_method}.json"


def filter_incomplete_parameter_sets(parameter_sets, results_dir="results"):
    """
    Filter parameter sets to only include those that haven't been completed yet.
    
    Args:
        parameter_sets: list of parameter dictionaries
        results_dir: directory containing result files
    
    Returns:
        list: List of parameter dictionaries that haven't been completed
    """
    incomplete_sets = []
    
    for params in parameter_sets:
        # Generate expected filename using the helper function
        filename = generate_variational_cooling_filename(
            params['system_qubits'], params['bath_qubits'], params['J'], params['h'],
            params['num_sweeps'], params['single_qubit_gate_noise'], 
            params['two_qubit_gate_noise'], params['training_method']
        )
        filepath = os.path.join(results_dir, filename)
        
        if not os.path.exists(filepath):
            incomplete_sets.append(params)
    
    return incomplete_sets


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
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate cluster jobs for data_collection.py')
    parser.add_argument('--study-type', choices=['original', 'alternative'], default='original',
                       help='Type of study: original (comprehensive sweep) or alternative (training methods)')
    parser.add_argument('--output-dir', default=None, help='Override output directory')
    parser.add_argument('--job-name', default=None, help='Override job name')
    parser.add_argument('--filter-completed', default=True, action='store_true',
                       help='Only generate jobs for parameter sets that haven\'t been completed yet')
    
    args = parser.parse_args()
    
    # Set output directory and job name based on study type
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f"jobs_{args.study_type}"
    
    if args.job_name:
        job_name = args.job_name
    else:
        job_name = f"{args.study_type}_data_collection"
    
    print(f"Generating cluster jobs for {args.study_type} data_collection.py study...")
    print(f"Output directory: {output_dir}")
    print("Each parameter set will run in a separate job")
    
    # Check if jobs directory exists and warn user
    if os.path.exists(output_dir):
        print(f"Warning: Jobs directory '{output_dir}' already exists.")
        print("This will overwrite existing job files. Press Ctrl+C to cancel or Enter to continue...")
        try:
            input()
        except KeyboardInterrupt:
            print("Operation cancelled.")
            exit(0)
        print("Continuing with job generation...")
    
    print()
    
    # Get all parameter sets
    print(f"Generating {args.study_type} parameter sets...")
    all_parameter_sets = create_parameter_sets(args.study_type)
    print(f"Total parameter sets: {len(all_parameter_sets)}")
    
    # Filter out completed parameter sets if requested
    if args.filter_completed:
        print(f"\nFiltering out completed parameter sets...")
        parameter_sets_to_run = filter_incomplete_parameter_sets(all_parameter_sets)
        
        if len(parameter_sets_to_run) == 0:
            print("All parameter sets have been completed! No jobs to generate.")
            sys.exit(0)
        else:
            print(f"Filtered to {len(parameter_sets_to_run)} incomplete parameter sets.")
    else:
        parameter_sets_to_run = all_parameter_sets
    
    # Print summary of parameter combinations
    config = STUDY_CONFIGS[args.study_type]
    print(f"\n{args.study_type.title()} parameter combinations summary:")
    print(f"Description: {config['description']}")
    print("=" * 50)
    
    # Summary by J,h values
    for J, h in config['j_h_list']:
        count = sum(1 for params in all_parameter_sets if params['J'] == J and params['h'] == h)
        print(f"J={J}, h={h}: {count} combinations")
    
    # Summary by system size
    for sys_qubits, bath_qubits in config['system_sizes']:
        count = sum(1 for params in all_parameter_sets 
                   if params['system_qubits'] == sys_qubits and params['bath_qubits'] == bath_qubits)
        print(f"{sys_qubits}+{bath_qubits} qubits: {count} combinations")
    
    # Summary by training methods
    for training_method in config['training_methods']:
        count = sum(1 for params in all_parameter_sets if params['training_method'] == training_method)
        print(f"Training method '{training_method}': {count} combinations")
    
    # Summary by noise levels
    for noise_factor in config['noise_factors']:
        count = sum(1 for params in all_parameter_sets 
                   if abs(params['single_qubit_gate_noise'] - BASE_SINGLE_QUBIT_NOISE * noise_factor) < 1e-6)
        print(f"Noise factor {noise_factor:.1f}: {count} combinations")
    
    # Summary of shot allocation
    print("\nShot allocation by system size:")
    print("-" * 30)
    for sys_qubits, bath_qubits in config['system_sizes']:
        current_total = sys_qubits + bath_qubits
        scaling_factor = BASE_TOTAL_SYSTEM_SIZE / current_total
        if current_total <= 12+6:
            shots = int(BASE_SHOTS_LARGE * scaling_factor)
        else:
            shots = int(BASE_SHOTS_SMALL * scaling_factor)
        print(f"{sys_qubits}+{bath_qubits} qubits: {shots} shots (scaling factor: {scaling_factor:.2f})")
    
    print("=" * 50)
    
    # Generate job files
    print(f"\nGenerating job files for {len(parameter_sets_to_run)} parameter combinations...")
    job_scripts = generate_jobs(
        parameter_sets_to_run, 
        output_dir=output_dir,
        job_name=job_name,
        wall_time=WALL_TIME,
        memory=MEMORY,
        cores=CORES,
        queue=QUEUE
    )
    
    # Create submit all script
    submit_script = create_submit_all_script(job_scripts, output_dir)
    print(f"Submit script: {submit_script}")
    
    # Create monitor script
    monitor_script = create_monitor_script(job_scripts, output_dir)
    print(f"Monitor script: {monitor_script}")
    
    print(f"\nTo submit all jobs:")
    print(f"  {submit_script}")
    print(f"\nTo monitor jobs:")
    print(f"  {monitor_script}")
    print(f"\nTo submit individual jobs:")
    for job_script in job_scripts:
        print(f"  bsub < {job_script}")
    
    print(f"\nTotal jobs created: {len(job_scripts)}")
    print(f"Total parameter combinations: {len(parameter_sets_to_run)}")
    if args.filter_completed:
        print(f"Total original parameter sets: {len(all_parameter_sets)}")
        print(f"Completed parameter sets: {len(all_parameter_sets) - len(parameter_sets_to_run)}")
    print("Each parameter set runs in its own job")
