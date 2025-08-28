#!/usr/bin/env python3
import quimb.tensor as qtn
import quimb.tensor.tensor_builder as qtn_builder
import quimb as qu
import numpy as np
from quimb.tensor.tensor_1d import MatrixProductState

def calculate_ising_ground_state(N, J, h, bond_dim=30, max_iter=200, cyclic=False):
    """
    Calculate the ground state energy of the transverse field Ising model.
    
    Parameters:
    -----------
    N : int
        Number of spins
    J : float
        Coupling strength
    h : float
        Transverse field strength
    bond_dim : int
        Bond dimension for DMRG
    max_iter : int
        Maximum number of DMRG iterations
    
    Returns:
    --------
    tuple
        (ground_state_energy, ground_state_mps)
    """
    
    # print(f"Building transverse field Ising model with:")
    # print(f"  N = {N} spins")
    # print(f"  J = {J}")
    # print(f"  h = {h}")
    # print(f"  Open boundary conditions")
    # print(f"  Bond dimension = {bond_dim}")
    # print()
    
    # Build the MPO Hamiltonian
    H = qtn_builder.MPO_ham_ising(L=N, j=-J*4, bx=h*2, S=1 / 2, cyclic=cyclic)
    
    # print(f"Hamiltonian shape: {H.shape}")
    # print(f"Number of MPO tensors: {len(H.tensors)}")
    # print()
    
    # Create initial random MPS
    psi0 = qtn.MPS_rand_state(N, bond_dim=bond_dim)
    
    # print("Starting DMRG optimization...")
    
    # Use DMRG to find the ground state
    # Create DMRG solver
    dmrg = qtn.DMRG2(H, bond_dims=bond_dim, p0=psi0)
    
    # Run DMRG optimization
    dmrg.solve(tol=1e-10, max_sweeps=max_iter)
    
    ground_state_energy = dmrg.energy
    ground_state_mps = dmrg.state
    
    # print(f"Ground state energy: {ground_state_energy:.8f}")
    
    return ground_state_energy, ground_state_mps


def calculate_spin_correlations(mps, max_distance=None):
    """
    Calculate σᶻσᶻ correlations in the MPS ground state.
    
    Parameters:
    -----------
    mps : MatrixProductState
        The ground state MPS
    max_distance : int, optional
        Maximum distance to calculate correlations for (default: N//2)
    
    Returns:
    --------
    dict
        Dictionary containing distances and correlation values
    """
    N = mps.nsites
    
    if max_distance is None:
        max_distance = N // 2
    
    # Create Z operator
    Sz = qu.pauli('Z')
    
    # Calculate correlations from center outward
    center = N // 2
    correlations = []
    distances = []
    
    # Start with distance 0 (i=j=center)
    i = center
    j = center
    
    while (i > 0) or (j < N - 1):
        # Calculate correlation for current (i, j) pair
        if i == j:
            # For i == j, σz_i σz_j = I, so correlation is 1
            corr_val = 1.0
        else:
            # Calculate connected correlation function using correlation method
            corr_val = mps.correlation(Sz, i, j)
        
        # Calculate distance between sites
        distance = abs(j - i)
        
        correlations.append(corr_val)
        distances.append(distance)
        
        # Move outward: decrease i, then increase j
        if i > 0:
            i -= 1
        if j < N - 1:
            j += 1

    return {
        'distances': distances,
        'correlations': correlations,
        'center': center,
        'nsites': N
    }


def calculate_zz_correlations(mps):
    """
    Calculate ZZ correlations for all pairs of qubits.
    
    Parameters:
    -----------
    mps : MatrixProductState
        The ground state MPS
    
    Returns:
    --------
    dict
        Dictionary containing ZZ correlations and qubit pairs
    """
    N = mps.nsites
    
    # Create Z operator for correlations
    Sz = qu.pauli('Z')
    
    # Calculate ZZ correlations for all pairs
    ZZ_correlations = []
    ZZ_pairs = []
    
    for i in range(N):
        for j in range(i+1, N):  # Only upper triangular to avoid duplicates
            # Calculate <Z_i Z_j> using correlation method
            ZZ_val = mps.correlation(Sz, i, j)
            ZZ_correlations.append(ZZ_val)
            ZZ_pairs.append((i, j))
    
    return {
        'ZZ_correlations': ZZ_correlations,
        'ZZ_pairs': ZZ_pairs,
        'nsites': N
    }


def main():
    """Main function to run the calculation."""
    
    print("=" * 60)
    print("Transverse Field Ising Model Ground State Calculation")
    print("=" * 60)
    print()
    
    try:
        # Calculate ground state energy and MPS
        energy, mps = calculate_ising_ground_state(N=28, J=0.45, h=0.55, bond_dim=30, max_iter=200, cyclic=False)
        print(f"Ground state energy: {energy:.8f}")
        
        # Calculate spin correlations
        print("\nCalculating σᶻσᶻ correlations...")
        corr_results = calculate_spin_correlations(mps)
        
        print(f"System size: {corr_results['nsites']} qubits")
        print(f"Center qubit: {corr_results['center']}")
        print(f"Number of correlations: {len(corr_results['correlations'])}")
        print()
        
        print("Distance | Correlation")
        print("-" * 25)
        for dist, corr in zip(corr_results['distances'], corr_results['correlations']):
            print(f"{dist:8d} | {corr:10.6f}")

    except Exception as e:
        print(f"Error during calculation: {e}")
        print("This might be due to insufficient bond dimension or convergence issues.")
        print("Try increasing the bond dimension or max iterations.")


if __name__ == "__main__":
    main()