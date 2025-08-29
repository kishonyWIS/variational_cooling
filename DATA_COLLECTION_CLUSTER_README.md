# Data Collection Cluster Workflow

This directory contains scripts to run `data_collection.py` on a cluster with different parameter combinations.

## Overview

The workflow consists of:
1. **Parameter Generation**: `cluster_job_generator_data_collection.py` creates parameter combinations
2. **Job Creation**: Generates LSF job scripts for the cluster (one job per parameter set)
3. **Execution**: Each job runs `data_collection.py` directly with its specific parameters
4. **Monitoring**: Scripts to monitor job progress

## Parameter Combinations

The system will run the following parameter combinations:

### System Sizes
- (4, 2) - 4 system qubits + 2 bath qubits
- (8, 4) - 8 system qubits + 4 bath qubits  
- (12, 6) - 12 system qubits + 6 bath qubits
- (16, 8) - 16 system qubits + 8 bath qubits
- (20, 10) - 20 system qubits + 10 bath qubits
- (24, 12) - 24 system qubits + 12 bath qubits
- (28, 14) - 28 system qubits + 14 bath qubits

### J, h Values
- (0.4, 0.6)
- (0.45, 0.55)
- (0.55, 0.45)
- (0.6, 0.4)

### Noise Levels
- (0.001, 0.01) × [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
- This gives noise ranges: (0.0, 0.0) to (0.001, 0.01)

### Fixed Parameters
- p = 3 (HVA layers per sweep)
- num_sweeps = 12
- training_method = 'energy'
- initial_state = 'zeros'
- bond_dimensions = [32, 64]
- num_shots = 100
- open_boundary = 1

**Total combinations**: 7 × 4 × 11 = 308 parameter sets

## Quick Start

### 1. Generate Job Files
```bash
python3 cluster_job_generator_data_collection.py
```

This will create:
- `jobs/` directory with LSF job scripts
- `jobs/job_001.lsf`, `jobs/job_002.lsf`, etc. (one per parameter set)
- `jobs/params_001.json`, `jobs/params_002.json`, etc. (one per parameter set)
- `jobs/submit_all.sh` - script to submit all jobs
- `jobs/monitor_jobs.sh` - script to monitor job status

### 2. Submit Jobs
```bash
# Submit all jobs at once
./jobs/submit_all.sh

# Or submit individual jobs
bsub < jobs/job_001.lsf
bsub < jobs/job_002.lsf
# etc.
```

### 3. Monitor Progress
```bash
# Check job status
./jobs/monitor_jobs.sh

# Or use LSF commands directly
bjobs                    # List all jobs
bjobs | grep data_collection  # List your jobs
```

## Job Configuration

### Default Settings
- **Jobs per file**: 1 parameter set per job (308 total jobs)
- **Wall time**: 48 hours per job
- **Memory**: 16GB per job
- **Queue**: "normal" (modify in script for your cluster)

### Customization
Edit the configuration section in `cluster_job_generator_data_collection.py`:

```python
# Configuration
WALL_TIME = "48:00"      # Wall time limit
MEMORY = "16GB"          # Memory requirement
QUEUE = "normal"         # Queue name (change for your cluster)
```

## Output Files

Each job will generate:
- **Variational cooling data**: `variational_cooling_data_sys{N}_bath{M}_J{J}_h{h}_sweeps{S}_noise{N1}_{N2}.json`
- **Ground state data**: `ground_state_data_sys{N}_J{J}_h{h}.json`

## Cluster Requirements

### Software Dependencies
- Python 3.7+
- Qiskit
- NumPy
- Other dependencies from `requirements.txt`

### Module Loading
Uncomment and modify the module loading section in the job scripts:
```bash
# module load python/3.9
# module load qiskit
```

### Queue Configuration
Change the queue name in `cluster_job_generator_data_collection.py` to match your cluster:
```python
QUEUE = "your_queue_name"  # e.g., "normal", "long", "gpu", etc.
```

## Troubleshooting

### Common Issues

1. **Memory errors**: Increase `MEMORY` in the configuration
2. **Timeouts**: Increase `WALL_TIME` for larger system sizes
3. **Queue errors**: Check available queues with `bqueues`
4. **Module errors**: Ensure required modules are loaded

### Debugging

- Check job output files: `jobs/job_001.out`, `jobs/job_001.err`
- Use `bpeek <job_id>` to see real-time output
- Check `results/` directory for generated files

### Performance Tips

- Larger system sizes (28+ qubits) may need more memory/time
- Each parameter set runs independently, so jobs can be distributed across cluster nodes
- Monitor cluster usage with `bhosts` and `bqueues`

## File Structure

```
.
├── cluster_job_generator_data_collection.py  # Main job generator
├── data_collection.py                        # Core data collection module (modified for CLI)
├── jobs/                                     # Generated job files
│   ├── job_001.lsf                          # One job per parameter set
│   ├── job_002.lsf
│   ├── params_001.json                      # One parameter file per job
│   ├── params_002.json
│   ├── submit_all.sh
│   └── monitor_jobs.sh
├── results/                                  # Output directory
└── DATA_COLLECTION_CLUSTER_README.md         # This file
```

## Example Workflow

```bash
# 1. Generate jobs
python3 cluster_job_generator_data_collection.py

# 2. Review generated files
ls -la jobs/
cat jobs/params_001.json

# 3. Submit jobs
./jobs/submit_all.sh

# 4. Monitor progress
./jobs/monitor_jobs.sh

# 5. Check results
ls -la results/
```

## Support

For issues or questions:
1. Check job output files for error messages
2. Verify cluster configuration and queue settings
3. Ensure all dependencies are properly installed
4. Check available cluster resources with `bhosts` and `bqueues`
