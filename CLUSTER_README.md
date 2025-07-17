# Variational Cooling Cluster Job System

This directory contains scripts to run variational cooling simulations on a cluster using LSF (Load Sharing Facility) job scheduler.

## Overview

The system splits parameter sets into separate jobs that can run in parallel on the cluster. Each job processes a subset of parameter combinations and saves results to individual CSV files.

## Files

- `variational_cooling_mps_simulation.py` - Main simulation script (modified to accept command line arguments)
- `cluster_job_generator.py` - Generates LSF job scripts and parameter files
- `combine_results.py` - Combines results from all jobs into a single file
- `CLUSTER_README.md` - This file

## Quick Start

### 1. Generate Job Files

```bash
python3 cluster_job_generator.py
```

This will:
- Create a `jobs/` directory
- Generate LSF job scripts (`job_001.lsf`, `job_002.lsf`, etc.)
- Create parameter files (`params_001.json`, `params_002.json`, etc.)
- Create submission and monitoring scripts

### 2. Submit Jobs

```bash
# Submit all jobs at once
./jobs/submit_all.sh

# Or submit individual jobs
bsub < jobs/job_001.lsf
bsub < jobs/job_002.lsf
# etc.
```

### 3. Monitor Jobs

```bash
# Check job status
./jobs/monitor_jobs.sh

# Or use LSF commands directly
bjobs                    # List all jobs
bjobs | grep variational_cooling  # List your jobs
```

### 4. Combine Results

```bash
# After all jobs complete, combine results
python3 combine_results.py --analyze
```

## Configuration

### Modify Parameter Sets

Edit the `create_example_parameter_sets()` function in `variational_cooling_mps_simulation.py` to define your parameter combinations:

```python
def create_example_parameter_sets():
    parameter_sets = []
    
    # Example: vary system size and number of sweeps
    for system_qubits, bath_qubits in [(10, 5), (20, 10), (30, 15)]:
        for num_sweeps in [1, 2, 3, 5]:
            for noise_factor in [1.0, 2.0, 5.0]:
                parameter_sets.append({
                    'energy_density_atol': 0.01,
                    'system_qubits': system_qubits,
                    'bath_qubits': bath_qubits,
                    'J': 0.4,
                    'h': 0.6,
                    'p': 3,
                    'num_sweeps': num_sweeps,
                    'single_qubit_gate_noise': 0.0003 * noise_factor,
                    'two_qubit_gate_noise': 0.003 * noise_factor,
                    'max_timeout_minutes': 30,
                    'max_bond_dim': 64,
                    'training_method': 'energy',
                    'initial_state': 'zeros'
                })
    
    return parameter_sets
```

### Modify Job Configuration

Edit the configuration section in `cluster_job_generator.py`:

```python
# Configuration
JOBS_PER_FILE = 2          # Parameter sets per job
OUTPUT_DIR = "jobs"        # Directory for job files
JOB_NAME = "variational_cooling"
WALL_TIME = "24:00"        # Wall clock time limit
MEMORY = "8GB"            # Memory requirement
CORES = 1                 # Number of cores
QUEUE = "normal"          # Queue name
```

### Cluster-Specific Modifications

You may need to modify the job scripts for your specific cluster:

1. **Module loading**: Uncomment and modify module load commands in the job scripts
2. **Queue names**: Change `QUEUE = "normal"` to your cluster's queue names
3. **Resource limits**: Adjust `WALL_TIME`, `MEMORY`, and `CORES` based on your cluster's capabilities
4. **Python environment**: Ensure the correct Python environment is available

## Job Script Structure

Each job script (`job_XXX.lsf`) contains:

```bash
#!/bin/bash
#BSUB -J variational_cooling_001    # Job name
#BSUB -q normal                     # Queue
#BSUB -W 24:00                     # Wall time
#BSUB -M 8GB                       # Memory
#BSUB -n 1                         # Cores
#BSUB -o jobs/job_001.out          # Output file
#BSUB -e jobs/job_001.err          # Error file

# Load modules (modify for your cluster)
# module load python/3.9
# module load qiskit

# Run simulation
python3 variational_cooling_mps_simulation.py --job-id 1 --param-file jobs/params_001.json --output-dir results
```

## Output Structure

```
results/
├── results_job_001.csv    # Results from job 1
├── results_job_002.csv    # Results from job 2
└── ...

combined_results.csv       # Combined results (after running combine_results.py)
```

## Monitoring and Debugging

### Check Job Status
```bash
bjobs                    # All jobs
bjobs | grep PEND        # Pending jobs
bjobs | grep RUN         # Running jobs
bjobs | grep DONE        # Completed jobs
```

### Check Job Output
```bash
# Check output files
cat jobs/job_001.out
cat jobs/job_001.err

# Monitor job progress
tail -f jobs/job_001.out
```

### Kill Jobs
```bash
# Kill specific job
bkill <job_id>

# Kill all your variational cooling jobs
bkill $(bjobs | grep variational_cooling | awk '{print $1}')
```

## Troubleshooting

### Common Issues

1. **Module not found**: Modify module load commands in job scripts
2. **Python path issues**: Ensure PYTHONPATH includes your code directory
3. **Memory/time limits**: Increase WALL_TIME or MEMORY in job configuration
4. **Queue issues**: Check available queues with `bqueues`

### Debug Mode

Run a single job interactively to debug:
```bash
# Submit job in interactive mode
bsub -I -q normal -W 1:00 -M 4GB -n 1 bash jobs/job_001.lsf
```

### Test Locally

Test parameter sets locally before submitting to cluster:
```bash
python3 variational_cooling_mps_simulation.py --param-file jobs/params_001.json --verbose
```

## Results Analysis

After combining results, you can analyze them:

```bash
# Basic combination
python3 combine_results.py

# Combine and analyze
python3 combine_results.py --analyze

# Custom analysis
python3 combine_results.py --output-file my_results.csv --analyze
```

The analysis provides:
- Summary by system size
- Summary by number of sweeps  
- Summary by noise level
- Convergence statistics
- Runtime statistics 