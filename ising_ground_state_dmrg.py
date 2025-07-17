#!/usr/bin/env python3
import quimb.tensor as qtn
import quimb.tensor.tensor_builder as qtn_builder
import numpy as np

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
    float
        Ground state energy
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
    
    # print(f"Ground state energy: {ground_state_energy:.8f}")
    
    return ground_state_energy

def main():
    """Main function to run the calculation."""
    
    print("=" * 60)
    print("Transverse Field Ising Model Ground State Calculation")
    print("=" * 60)
    print()
    
    try:
        # Calculate ground state energy
        energy = calculate_ising_ground_state(N=10, J=0.4, h=0.6, bond_dim=30, max_iter=200, cyclic=False)
        print(f"Ground state energy: {energy:.8f}")

    except Exception as e:
        print(f"Error during calculation: {e}")
        print("This might be due to insufficient bond dimension or convergence issues.")
        print("Try increasing the bond dimension or max iterations.")

if __name__ == "__main__":
    main()