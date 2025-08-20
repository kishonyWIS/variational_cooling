# Transverse Field Ising Model Ground State Calculation

This repository contains a Python script to calculate the ground state energy of the transverse field Ising model using the quimb tensor network library, with additional functionality for variational cooling and σᶻσᶻ correlation function analysis.

## Model Parameters

- **N = 28** spins
- **J = 0.4** (coupling strength)
- **h = 0.6** (transverse field strength)
- **Open boundary conditions**

## Features

### Core Functionality
1. **Ground State Calculation**: Uses DMRG to find exact ground state energies
2. **Variational Cooling**: Implements HVA circuits for quantum state preparation
3. **σᶻσᶻ Correlation Functions**: Calculates spin-spin correlations along the chain
4. **Cluster Computing**: Supports parallel execution on HPC clusters
5. **Noise Analysis**: Studies effects of quantum gate errors

### Correlation Functions
The code can calculate σᶻσᶻ correlation functions between system qubits, symmetrically around the center of the chain:
- **Distance-based correlations**: Measures correlations between qubits at different distances
- **Symmetric placement**: Correlations are calculated around the center qubit
- **Configurable**: Can be enabled/disabled via command line arguments

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Ground State Calculation
```bash
python ising_ground_state.py
```

### Variational Cooling with Correlations
```bash
# Include correlation function calculations
python variational_cooling_mps_simulation.py --include-correlations --analyze-correlations

# Run on cluster with correlations
python cluster_job_generator.py
```

### Command Line Options
- `--include-correlations`: Enable σᶻσᶻ correlation function calculations
- `--analyze-correlations`: Analyze and plot correlation functions
- `--bond-dims`: Specify bond dimensions for MPS simulation
- `--energy-density-atol`: Set energy density tolerance

## What the script does

1. **Builds the MPO Hamiltonian**: Uses `quimb.tensor.tensor_builder.MPO_ham_ising()` to construct the matrix product operator representation of the transverse field Ising Hamiltonian
2. **Initializes a random MPS**: Creates a random matrix product state as the initial guess
3. **Performs DMRG optimization**: Uses the Davidson algorithm (`eigsh`) to find the ground state energy
4. **Calculates correlations**: When enabled, computes σᶻσᶻ correlation functions between system qubits
5. **Returns the result**: Prints the calculated ground state energy and correlation data

## Technical Details

- The Hamiltonian is constructed as an MPO (Matrix Product Operator)
- DMRG (Density Matrix Renormalization Group) is used for ground state search
- Default bond dimension is 20 (can be adjusted in the function parameters)
- Open boundary conditions are used (not periodic)
- Correlation functions are calculated symmetrically around the chain center

## Expected Output

The script will output:
- Model parameters being used
- Hamiltonian information
- Progress during optimization
- Final ground state energy
- σᶻσᶻ correlation functions (when enabled)
- Correlation analysis plots (when verbose mode is on)

## Dependencies

- `quimb`: Tensor network library for quantum many-body physics
- `numpy`: Numerical computing library
- `scipy`: Scientific computing library (used by quimb)
- `qiskit`: Quantum computing framework
- `matplotlib`: Plotting library 