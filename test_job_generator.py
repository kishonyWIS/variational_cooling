#!/usr/bin/env python3
"""
Test script for the job generator.
"""

import os
import tempfile
import shutil
from cluster_job_generator_data_collection import create_parameter_sets, generate_jobs

def test_job_generation():
    """Test that the job generator creates the expected number of jobs."""
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Testing in temporary directory: {temp_dir}")
        
        # Generate parameter sets
        print("Generating parameter sets...")
        params = create_parameter_sets()
        print(f"Generated {len(params)} parameter sets")
        
        # Test first few parameter sets
        print("\nFirst 3 parameter sets:")
        for i, p in enumerate(params[:3]):
            print(f"  {i+1}: sys={p['system_qubits']}+{p['bath_qubits']}, J={p['J']}, h={p['h']}, noise=({p['single_qubit_gate_noise']:.6f}, {p['two_qubit_gate_noise']:.6f})")
        
        # Generate jobs
        print(f"\nGenerating jobs in {temp_dir}...")
        job_scripts = generate_jobs(params, output_dir=temp_dir)
        print(f"Generated {len(job_scripts)} job scripts")
        
        # Check that we have the right number of jobs
        expected_jobs = 7 * 4 * 11  # system_sizes × J_h_values × noise_levels
        assert len(job_scripts) == expected_jobs, f"Expected {expected_jobs} jobs, got {len(job_scripts)}"
        
        # Check that job files exist
        for job_script in job_scripts[:3]:  # Check first 3
            assert os.path.exists(job_script), f"Job script {job_script} not found"
            print(f"  ✓ {os.path.basename(job_script)} exists")
        
        # Check that parameter files exist
        for i in range(1, 4):  # Check first 3
            param_file = os.path.join(temp_dir, f"params_{i:03d}.json")
            assert os.path.exists(param_file), f"Parameter file {param_file} not found"
            print(f"  ✓ {os.path.basename(param_file)} exists")
        
        print(f"\n✓ All tests passed! Generated {len(job_scripts)} jobs correctly.")
        
        # Show job distribution
        print(f"\nJob distribution:")
        print(f"  Total jobs: {len(job_scripts)}")
        print(f"  Expected: {expected_jobs}")
        print(f"  Each parameter set gets its own job: ✓")

if __name__ == "__main__":
    test_job_generation()
