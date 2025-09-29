from ising_ground_state_dmrg import calculate_ising_ground_state
from ising_ground_state_dmrg import calculate_zz_correlations
import time
from typing import Dict, Any
import quimb.tensor as qtn


def compute_ground_state_observables(system_qubits: int, J: float, h: float, 
                                    bond_dim: int = 50, max_iter: int = 200, psi0=None) -> Dict[str, Any]:
    """
    Compute ground state observables using DMRG.
    
    Args:
        system_qubits: number of system qubits
        J: Ising coupling strength
        h: transverse field strength
        bond_dim: bond dimension for DMRG
        max_iter: maximum number of DMRG iterations
    
    Returns:
        dict: ground state observables and metadata
    """
    
    print(f"Computing ground state observables for {system_qubits} qubits, J={J}, h={h}")
    

    # Calculate ground state using DMRG
    E0, ground_state_mps = calculate_ising_ground_state(
        N=system_qubits, J=J, h=h, bond_dim=bond_dim, max_iter=max_iter, cyclic=False, psi0=psi0
    )
    
    # Calculate ZZ correlations using the simplified DMRG function
    from ising_ground_state_dmrg import calculate_zz_correlations
    correlation_results = calculate_zz_correlations(ground_state_mps)
    
    # Calculate expectation values
    ground_state_results = {
        'ground_state_energy': float(E0),
        'bond_dim_used': bond_dim,
        'observables': {}
    }
    
    # Process ZZ correlations
    for i, (qubit_i, qubit_j) in enumerate(correlation_results['ZZ_pairs']):
        label = f"ZZ_{qubit_i}_{qubit_j}"
        ground_state_results['observables'][label] = {
            'value': float(correlation_results['ZZ_correlations'][i]),
            'std_error': 0.0  # DMRG gives exact values, no statistical uncertainty
        }
    
    # Create comprehensive ground state data
    ground_state_data = {
        'metadata': {
            'system_qubits': system_qubits,
            'J': J,
            'h': h,
            'bond_dim': bond_dim,
            'max_iter': max_iter,
            'computation_timestamp': time.time()
        },
        'ground_state_results': ground_state_results
    }
    
    print(f"Ground state energy: {E0:.8f}")
    print(f"Calculated {len(ground_state_results['observables'])} observables")
        
    # print correlation between N//2 and N//2+1
    print(f"Correlation between {N//2} and {N//2+1}: {ground_state_results['observables'][f'ZZ_{N//2}_{N//2+1}']['value']}")

    return ground_state_data, ground_state_mps


if __name__ == "__main__":
    N = 28
    ground_state_data, ground_state_mps = compute_ground_state_observables(system_qubits=N, J=1., h=0., bond_dim=50, max_iter=200, psi0=None)
    ground_state_data, ground_state_mps = compute_ground_state_observables(system_qubits=N, J=0.59, h=0.41, bond_dim=50, max_iter=200, psi0=None)
    # first run with random initial state for J=0.6, h=0.4
    ground_state_data, ground_state_mps = compute_ground_state_observables(system_qubits=N, J=0.6, h=0.4, bond_dim=50, max_iter=200, psi0=None)
    # then run with for J=0.55, h=0.45
    ground_state_data, ground_state_mps = compute_ground_state_observables(system_qubits=N, J=0.55, h=0.45, bond_dim=50, max_iter=200, psi0=None)
    # then rerun for J=0.6, h=0.4 using the ground state MPS as initial state
    ground_state_data, ground_state_mps = compute_ground_state_observables(system_qubits=N, J=0.6, h=0.4, bond_dim=50, max_iter=200, psi0=ground_state_mps)
    psi0 = qtn.MPS_computational_state(binary="01" * (N//2))
    ground_state_data, ground_state_mps = compute_ground_state_observables(system_qubits=N, J=0.6, h=0.4, bond_dim=50, max_iter=200, psi0=psi0)
    psi0 = qtn.MPS_computational_state(binary="01" * (N//2))
    ground_state_data, ground_state_mps = compute_ground_state_observables(system_qubits=N, J=0.55, h=0.45, bond_dim=50, max_iter=200, psi0=psi0)