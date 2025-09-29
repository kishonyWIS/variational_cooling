#!/usr/bin/env python3
import quimb.tensor as qtn
import quimb.tensor.tensor_builder as qtn_builder
import quimb as qu
import numpy as np
from quimb.tensor.tensor_1d import MatrixProductState

def calculate_ising_ground_state(N, J, h, bond_dim=30, max_iter=200, cyclic=False, psi0=None):
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
    if psi0 is None:
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


def calculate_raw_zz_correlation(mps, i, j):
    """
    Calculate raw ZZ correlation <Z_i Z_j> (not connected correlation).
    
    Parameters:
    -----------
    mps : MatrixProductState
        The ground state MPS
    i : int
        First site index
    j : int
        Second site index
    
    Returns:
    --------
    float
        Raw correlation <Z_i Z_j>
    """
    from quimb.tensor.tensor_1d import expec_TN_1D
    
    # Create Z operator
    Sz = qu.pauli('Z')
    
    # Calculate raw correlation <Z_i Z_j> by computing expectation value
    # of the product operator Z_i Z_j, similar to the correlation method
    # but without subtracting individual terms
    
    bra = mps.H
    
    # Apply Z operator to site i
    pA = mps.gate(Sz, i, contract=True)
    
    # Apply Z operator to site j (on the already modified state)
    pAB = pA.gate_(Sz, j, contract=True)
    
    # Compute expectation value <Z_i Z_j>
    ZZ_val = expec_TN_1D(bra, pAB)
    
    return ZZ_val


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
    
    # Calculate ZZ correlations for all pairs
    ZZ_correlations = []
    ZZ_pairs = []
    
    for i in range(N):
        for j in range(i+1, N):  # Only upper triangular to avoid duplicates
            # Calculate raw <Z_i Z_j> correlation
            ZZ_val = calculate_raw_zz_correlation(mps, i, j)
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
        
        print("Ground state calculation completed successfully!")

    except Exception as e:
        print(f"Error during calculation: {e}")
        print("This might be due to insufficient bond dimension or convergence issues.")
        print("Try increasing the bond dimension or max iterations.")


if __name__ == "__main__":
    main()