# Variational Cooling Code Refactoring

This document describes the refactored variational cooling simulation code that separates data collection from data analysis.

## Overview

The original `variational_cooling_mps_simulation.py` file has been refactored into a cleaner, more modular structure:

- **`data_collection.py`**: Handles all data collection from variational cooling simulations
- **`ising_ground_state_dmrg.py`**: Enhanced with additional observable calculation functions
- **`test_data_collection.py`**: Test script to verify the refactored code works correctly

## Key Changes

### 1. Separation of Concerns

- **Data Collection**: All simulation and measurement logic is now in `data_collection.py`
- **Data Analysis**: Analysis and plotting functions remain in the original file (can be further separated later)
- **Ground State Computation**: DMRG calculations are enhanced and separated

### 2. Enhanced Observable Measurements

The refactored code now measures observables at multiple points:
- **Initial State**: Before any cooling sweeps
- **After Each Sweep**: Measurements are taken after each complete sweep (before bath reset)

### 3. Comprehensive Observable Set

For each measurement point, the following observables are computed:
- **Single-qubit X**: `<X_i>` for all system qubits i
- **Single-qubit Z**: `<Z_i>` for all system qubits i  
- **Two-qubit ZZ**: `<Z_i Z_j>` for all pairs of system qubits (i, j)

### 4. Improved Data Storage

- **JSON Format**: Data is saved in structured JSON files for easy post-processing
- **Metadata**: Complete parameter information is stored with each dataset
- **Timestamps**: Each dataset includes collection timestamps
- **Multiple Bond Dimensions**: Results for different bond dimensions are stored together

## Usage

### Basic Data Collection

```python
from data_collection import collect_variational_cooling_data

# Collect data from variational cooling simulation
collected_data = collect_variational_cooling_data(
    system_qubits=10,
    bath_qubits=5,
    open_boundary=1,
    J=0.4,
    h=0.6,
    p=3,
    num_sweeps=3,
    single_qubit_gate_noise=0.0003,
    two_qubit_gate_noise=0.003,
    training_method='energy',
    initial_state='zeros',
    bond_dimensions=[32, 64],
    num_shots=100,
    output_dir='results'
)
```

### Ground State Computation

```python
from data_collection import compute_ground_state_observables

# Compute ground state observables using DMRG
ground_state_data = compute_ground_state_observables(
    system_qubits=10,
    J=0.4,
    h=0.6,
    bond_dim=50,
    max_iter=200
)
```

### Testing

Run the test suite to verify everything works:

```bash
python test_data_collection.py
```

## Data Structure

### Variational Cooling Data

The collected data has the following structure:

```json
{
  "metadata": {
    "system_qubits": 10,
    "bath_qubits": 5,
    "open_boundary": 1,
    "J": 0.4,
    "h": 0.6,
    "p": 3,
    "num_sweeps": 3,
    "single_qubit_gate_noise": 0.0003,
    "two_qubit_gate_noise": 0.003,
    "training_method": "energy",
    "initial_state": "zeros",
    "bond_dimensions": [32, 64],
    "num_shots": 100,
    "collection_timestamp": 1234567890.123,
    "observable_labels": ["X_0", "X_1", ..., "Z_0", "Z_1", ..., "ZZ_0_1", ...],
    "observable_types": ["single_X", "single_X", ..., "single_Z", "single_Z", ..., "correlation", ...]
  },
  "results": {
    "bond_dim_32": {
      "bond_dim": 32,
      "shots": 100,
      "initial_state_measurements": {
        "X_0": {"value": 0.123, "std": 0.045, "type": "single_X"},
        "Z_0": {"value": -0.234, "std": 0.056, "type": "single_Z"},
        "ZZ_0_1": {"value": 0.345, "std": 0.067, "type": "correlation"}
      },
      "sweep_measurements": {
        "sweep_0": {
          "X_0": {"value": 0.234, "std": 0.045, "type": "single_X"},
          "Z_0": {"value": -0.345, "std": 0.056, "type": "single_Z"},
          "ZZ_0_1": {"value": 0.456, "std": 0.067, "type": "correlation"}
        },
        "sweep_1": {
          "X_0": {"value": 0.345, "std": 0.045, "type": "single_X"},
          "Z_0": {"value": -0.456, "std": 0.056, "type": "single_Z"},
          "ZZ_0_1": {"value": 0.567, "std": 0.067, "type": "correlation"}
        }
      }
    },
    "bond_dim_64": {
      // Similar structure for bond dimension 64
    }
  }
}
```

### Ground State Data

```json
{
  "metadata": {
    "system_qubits": 10,
    "J": 0.4,
    "h": 0.6,
    "bond_dim": 50,
    "max_iter": 200,
    "computation_timestamp": 1234567890.123
  },
  "ground_state_results": {
    "ground_state_energy": -12.345678,
    "bond_dim_used": 50,
    "observables": {
      "X_0": {"value": 0.123, "std": 0.0, "type": "single_X"},
      "Z_0": {"value": -0.234, "std": 0.0, "type": "single_Z"},
      "ZZ_0_1": {"value": 0.345, "std": 0.0, "type": "correlation"}
    }
  }
}
```

## Benefits of Refactoring

1. **Modularity**: Clear separation between data collection and analysis
2. **Reusability**: Data collection functions can be used independently
3. **Maintainability**: Easier to modify and extend individual components
4. **Testing**: Isolated components are easier to test
5. **Data Quality**: More comprehensive observable measurements
6. **Storage**: Better organized data storage for post-processing

## Next Steps

The refactored code provides a solid foundation for:

1. **Data Analysis Module**: Create a separate module for analyzing collected data
2. **Plotting Functions**: Move plotting functions to a dedicated visualization module
3. **Parameter Studies**: Easier to run systematic parameter studies
4. **Cluster Execution**: Better suited for distributed computing

## Dependencies

- Qiskit (for quantum circuits and simulation)
- NumPy (for numerical operations)
- QuIMB (for DMRG calculations)
- Standard Python libraries (json, os, time, typing)

## File Organization

```
variational_cooling/
├── data_collection.py           # Main data collection module
├── ising_ground_state_dmrg.py  # Enhanced DMRG module
├── test_data_collection.py     # Test suite
├── REFACTOR_README.md          # This documentation
├── variational_cooling_mps_simulation.py  # Original file (for reference)
└── results/                    # Output directory for collected data
```
