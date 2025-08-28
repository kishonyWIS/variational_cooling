#!/usr/bin/env python3
"""
Debug script to understand the observable structure.
"""

from qiskit.quantum_info import SparsePauliOp
from variational_cooling_mps_simulation import pauli_sys_ZZ, pauli_sys_X, get_best_parameters


def debug_observables():
    """Debug the observable structure."""
    system_qubits = 4
    bath_qubits = 2
    J = 0.4
    h = 0.6
    p = 3
    
    # Create system Hamiltonian
    pauli_sys = pauli_sys_ZZ(system_qubits, 0, J, 1) + pauli_sys_X(system_qubits, 0, h)
    H_sys = SparsePauliOp.from_list(pauli_sys)
    
    print(f"System Hamiltonian:")
    print(f"  Number of terms: {len(pauli_sys)}")
    print(f"  First few terms:")
    for i, (pauli_str, coeff) in enumerate(pauli_sys[:3]):
        print(f"    {i}: {pauli_str} (coeff: {coeff})")
    
    # Create observables for the enlarged circuit
    H_observables = []
    for pauliop in pauli_sys:
        bt_str = (bath_qubits)*"I" + pauliop[0]
        H_observables.append((bt_str, pauliop[1]))
    
    all_observables = H_observables
    observables = SparsePauliOp.from_list(all_observables)
    
    print(f"\nEnlarged observables:")
    print(f"  Number of observables: {len(all_observables)}")
    print(f"  First few observables:")
    for i, (pauli_str, coeff) in enumerate(all_observables[:3]):
        print(f"    {i}: {pauli_str} (coeff: {coeff})")
        print(f"      Length: {len(pauli_str)}")
        print(f"      Non-I positions: {[j for j, p in enumerate(pauli_str) if p != 'I']}")
    
    print(f"\nSparsePauliOp:")
    print(f"  Number of qubits: {observables.num_qubits}")
    
    # Test individual observables
    for i, obs in enumerate(observables[:3]):
        print(f"\nObservable {i}:")
        print(f"  String representation: {obs}")
        print(f"  Number of qubits: {obs.num_qubits}")


if __name__ == "__main__":
    debug_observables()
