#!/usr/bin/env python3
"""
Small test script for σᶻσᶻ correlation functionality.
Tests with minimal parameters to avoid affecting main results.
"""

import os
import sys
from variational_cooling_mps_simulation import (
    fixed_bond_dimension_study, 
    pauli_sys_ZZ_correlations,
    analyze_correlations
)

def test_correlations():
    """Test correlation functionality with minimal parameters."""
    
    # Test parameters (small system, minimal sweeps)
    test_params = {
        'system_qubits': 8,
        'bath_qubits': 4,
        'J': 0.6,
        'h': 0.4,
        'num_sweeps': 10,  # Just 1 sweep for quick testing
        'p': 3,
        'single_qubit_gate_noise': 0.0,  # No noise for clean testing
        'two_qubit_gate_noise': 0.0,
        'training_method': 'energy',
        'initial_state': 'zeros',
        'bond_dimensions': [32],  # Test both bond dimensions

        'include_correlations': True
    }
    
    print("="*60)
    print("Testing σᶻσᶻ Correlation Functionality")
    print("="*60)
    print(f"System: {test_params['system_qubits']} + {test_params['bath_qubits']} qubits")
    print(f"Parameters: J={test_params['J']}, h={test_params['h']}")
    print(f"Sweeps: {test_params['num_sweeps']}")
    print(f"Bond dimensions: {test_params['bond_dimensions']}")
    print()
    
    # Test 1: Verify correlation generation
    print("Test 1: Correlation Generation")
    print("-" * 30)
    correlations, indices = pauli_sys_ZZ_correlations(
        test_params['system_qubits'], 
        test_params['bath_qubits']
    )
    print(f"Generated {len(correlations)} correlations")
    
    # Show first few correlations
    print("\nFirst 5 correlations:")
    for i, (pauli_str, coeff) in enumerate(correlations[:5]):
        idx_i, idx_j = indices[i]
        print(f"  {i}: {pauli_str} (coeff: {coeff}) -> Z_{idx_i}Z_{idx_j}")
    
    if len(correlations) > 5:
        print(f"  ... and {len(correlations) - 5} more")
    
    print()
    
    # Test 2: Run small simulation with correlations
    print("Test 2: Small Simulation with Correlations")
    print("-" * 40)
    
    # Use a separate test CSV file
    test_csv = "test_correlations_results.csv"
    
    try:
        # Remove test file if it exists
        if os.path.exists(test_csv):
            os.remove(test_csv)
        
        # Run the simulation
        results = fixed_bond_dimension_study(
            csv_file=test_csv,
            verbose=True,
            **test_params
        )
        
        print(f"\n✓ Simulation completed successfully!")
        print(f"Results saved to: {test_csv}")
        
        # Test 3: Verify CSV output
        print("\nTest 3: CSV Output Verification")
        print("-" * 35)
        
        if os.path.exists(test_csv):
            import pandas as pd
            df = pd.read_csv(test_csv)
            print(f"CSV loaded: {len(df)} rows")
            
            # Check for correlation columns
            correlation_cols = [col for col in df.columns if 'correlation_' in col]
            print(f"Correlation columns found: {len(correlation_cols)}")
            
            if correlation_cols:
                print("Sample correlation columns:")
                for col in correlation_cols[:5]:
                    print(f"  {col}")
                
                # Show correlation values for first row
                first_row = df.iloc[0]
                print(f"\nFirst row correlation values:")
                for i in range(len(correlations)):
                    val_col = f'correlation_{i}_value'
                    std_col = f'correlation_{i}_std'
                    if val_col in df.columns and std_col in df.columns:
                        val = first_row[val_col]
                        std = first_row[std_col]
                        print(f"  Correlation {i}: {val:.6f} ± {std:.6f}")
        
        # Clean up test file
        if os.path.exists(test_csv):
            os.remove(test_csv)
            print(f"\n✓ Test file cleaned up")
        
    except Exception as e:
        print(f"✗ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("Testing completed!")

if __name__ == "__main__":
    test_correlations()
