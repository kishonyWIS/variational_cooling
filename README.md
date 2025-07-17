# Transverse Field Ising Model Ground State Calculation

This repository contains a Python script to calculate the ground state energy of the transverse field Ising model using the quimb tensor network library.

## Model Parameters

- **N = 28** spins
- **J = 0.4** (coupling strength)
- **h = 0.6** (transverse field strength)
- **Open boundary conditions**

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the script to calculate the ground state energy:

```bash
python ising_ground_state.py
```

## What the script does

1. **Builds the MPO Hamiltonian**: Uses `quimb.tensor.tensor_builder.MPO_ham_ising()` to construct the matrix product operator representation of the transverse field Ising Hamiltonian
2. **Initializes a random MPS**: Creates a random matrix product state as the initial guess
3. **Performs DMRG optimization**: Uses the Davidson algorithm (`eigsh`) to find the ground state energy
4. **Returns the result**: Prints the calculated ground state energy

## Technical Details

- The Hamiltonian is constructed as an MPO (Matrix Product Operator)
- DMRG (Density Matrix Renormalization Group) is used for ground state search
- Default bond dimension is 20 (can be adjusted in the function parameters)
- Open boundary conditions are used (not periodic)

## Expected Output

The script will output:
- Model parameters being used
- Hamiltonian information
- Progress during optimization
- Final ground state energy

## Dependencies

- `quimb`: Tensor network library for quantum many-body physics
- `numpy`: Numerical computing library
- `scipy`: Scientific computing library (used by quimb) 