#!/usr/bin/env python3
"""
Test script for your specific parameter set.
Tests correlations with J_h_list, num_sweeps=12, noise_factor=0, system_size=(8,4)
"""

import os
import sys
from variational_cooling_mps_simulation import (
    fixed_bond_dimension_study, 
    pauli_sys_ZZ_correlations
)

def test_your_parameters():
    """Test with your specific parameter set."""
    
    # Your requested parameters
    J_h_list = [(0.6, 0.4), (0.55, 0.45), (0.45, 0.55), (0.4, 0.6)]
    num_sweeps = 12
    noise_factor = 0
    system_size = (28, 14)
    
    print("="*70)
    print("Testing Your Specific Parameter Set")
    print("="*70)
    print(f"System size: {system_size[0]} + {system_size[1]} qubits")
    print(f"Number of sweeps: {num_sweeps}")
    print(f"Noise factor: {noise_factor}")
    print(f"J, h combinations: {J_h_list}")
    print()
    
    # Test each J, h combination
    for J, h in J_h_list:
        print(f"Testing J={J}, h={h}")
        print("-" * 30)
        
        test_params = {
            'system_qubits': system_size[0],
            'bath_qubits': system_size[1],
            'J': J,
            'h': h,
            'num_sweeps': num_sweeps,
            'p': 3,
            'single_qubit_gate_noise': 0.0,  # noise_factor = 0
            'two_qubit_gate_noise': 0.0,
            'training_method': 'energy',
            'initial_state': 'zeros',
            'bond_dimensions': [32,64],  # Test both bond dimensions
            'energy_density_atol': 0.01,
            'include_correlations': True
        }
        
        # Use a separate test CSV file for each combination
        test_csv = f"test_your_params_J{J}_h{h}.csv"
        

        # Remove test file if it exists
        if os.path.exists(test_csv):
            os.remove(test_csv)
        
        # Run the simulation
        results = fixed_bond_dimension_study(
            csv_file=test_csv,
            verbose=True,
            **test_params
        )
        
        print(f"✓ Completed successfully!")
        print(f"Results saved to: {test_csv}")
        
        # Verify correlation columns
        if os.path.exists(test_csv):
            import pandas as pd
            df = pd.read_csv(test_csv)
            correlation_cols = [col for col in df.columns if 'correlation_' in col]
            print(f"  Correlation columns: {len(correlation_cols)}")
            
            # Show first few correlation values
            if len(df) > 0:
                first_row = df.iloc[0]
                print("  First few correlations:")
                for i in range(min(3, len(correlation_cols)//2)):
                    val_col = f'correlation_{i}_value'
                    if val_col in df.columns:
                        val = first_row[val_col]
                        print(f"    Correlation {i}: {val:.6f}")
                    
        print()
    
    print("="*70)
    print("Testing completed!")

if __name__ == "__main__":
    test_your_parameters()
