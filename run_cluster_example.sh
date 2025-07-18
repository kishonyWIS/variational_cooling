#!/bin/bash
# Example script demonstrating the complete cluster workflow

echo "=== Variational Cooling Cluster Job Example ==="
echo

# Step 1: Generate job files
echo "Step 1: Generating job files..."
python3 cluster_job_generator.py
echo

# Step 2: Show what was created
echo "Step 2: Generated files:"
ls -la jobs/
echo

# Step 3: Show job submission commands
echo "Step 3: To submit jobs, run one of:"
echo "  ./jobs/submit_all.sh                    # Submit all jobs"
echo "  bsub < jobs/job_001.lsf                 # Submit individual job"
echo "  bsub < jobs/job_002.lsf"
echo

# Step 4: Show monitoring commands
echo "Step 4: To monitor jobs:"
echo "  ./jobs/monitor_jobs.sh                  # Check status"
echo "  bjobs                                   # List all jobs"
echo "  bjobs | grep variational_cooling        # List your jobs"
echo

# Step 5: Show results combination
echo "Step 5: After jobs complete, combine results:"
echo "  python3 combine_results.py --analyze    # Combine and analyze"
echo

# Step 6: Show example parameter file
echo "Step 6: Example parameter file (jobs/params_001.json):"
if [ -f "jobs/params_001.json" ]; then
    head -20 jobs/params_001.json
else
    echo "Parameter file not found (run cluster_job_generator.py first)"
fi
echo

# Step 7: Show example job script
echo "Step 7: Example job script (jobs/job_001.lsf):"
if [ -f "jobs/job_001.lsf" ]; then
    head -15 jobs/job_001.lsf
else
    echo "Job script not found (run cluster_job_generator.py first)"
fi
echo

# Step 8: Show bond dimensions and energy density tolerance configuration
echo "Step 8: Bond dimensions and energy density tolerance configuration:"
echo "  Default bond dimensions: 32,64 (configured in cluster_job_generator.py)"
echo "  Default energy density tolerance: 0.01 (configured in cluster_job_generator.py)"
echo "  To test with different bond dimensions:"
echo "    python3 variational_cooling_mps_simulation.py --param-file jobs/params_001.json --bond-dims 16,32,64,128 --verbose"
echo "  To test with different energy density tolerance:"
echo "    python3 variational_cooling_mps_simulation.py --param-file jobs/params_001.json --energy-density-atol 0.005 --verbose"
echo

echo "=== Workflow Complete ==="
echo "Next steps:"
echo "1. Review generated files"
echo "2. Modify parameters in variational_cooling_mps_simulation.py if needed"
echo "3. Configure bond dimensions and energy density tolerance in cluster_job_generator.py if needed (default: 32,64, energy_density_atol=0.01)"
echo "4. Submit jobs using the commands above"
echo "5. Monitor progress"
echo "6. Combine results when complete" 