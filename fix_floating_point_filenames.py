#!/usr/bin/env python3
"""
Script to fix floating-point precision issues in variational cooling data filenames.
Renames files with problematic floating-point representations to clean versions.
"""

import os
import re
import shutil
from typing import Tuple, Optional


def normalize_float_for_filename(value, max_precision=6):
    """
    Normalize a float value for consistent filename representation.
    Handles floating-point precision issues by rounding to a reasonable precision
    and removing trailing zeros.
    
    Args:
        value: float value to normalize
        max_precision: maximum number of decimal places to consider
    
    Returns:
        str: normalized string representation
    """
    # Round to max_precision decimal places to handle floating-point errors
    rounded_value = round(value, max_precision)
    
    # Convert to string with max_precision decimal places
    str_value = f"{rounded_value:.{max_precision}f}"
    
    # Remove trailing zeros and decimal point if not needed
    str_value = str_value.rstrip('0').rstrip('.')
    
    return str_value


def clean_float_string(float_str: str) -> str:
    """
    Clean a float string by normalizing it to handle floating-point precision issues.
    
    Args:
        float_str: string representation of a float
        
    Returns:
        str: cleaned float string
    """
    try:
        float_val = float(float_str)
        return normalize_float_for_filename(float_val)
    except ValueError:
        return float_str


def parse_variational_cooling_filename(filename: str) -> Optional[dict]:
    """
    Parse a variational cooling data filename and extract parameters.
    
    Args:
        filename: filename to parse
        
    Returns:
        dict: parsed parameters or None if parsing fails
    """
    # Pattern to match variational cooling data filenames
    pattern = r"variational_cooling_data_sys(\d+)_bath(\d+)_J([\d.]+)_h([\d.]+)_sweeps(\d+)_noise([\d.e-]+)_([\d.e-]+)_method(\w+)\.json"
    
    match = re.match(pattern, filename)
    if not match:
        return None
    
    return {
        'system_qubits': int(match.group(1)),
        'bath_qubits': int(match.group(2)),
        'J': float(match.group(3)),
        'h': float(match.group(4)),
        'num_sweeps': int(match.group(5)),
        'single_qubit_noise': match.group(6),
        'two_qubit_noise': match.group(7),
        'training_method': match.group(8)
    }


def generate_clean_filename(params: dict) -> str:
    """
    Generate a clean filename from parameters.
    
    Args:
        params: dictionary with filename parameters
        
    Returns:
        str: clean filename
    """
    # Clean the noise values
    single_qubit_noise_clean = clean_float_string(params['single_qubit_noise'])
    two_qubit_noise_clean = clean_float_string(params['two_qubit_noise'])
    
    return f"variational_cooling_data_sys{params['system_qubits']}_bath{params['bath_qubits']}_J{params['J']}_h{params['h']}_sweeps{params['num_sweeps']}_noise{single_qubit_noise_clean}_{two_qubit_noise_clean}_method{params['training_method']}.json"


def fix_floating_point_filenames(results_dir: str = "results", dry_run: bool = True) -> Tuple[int, int]:
    """
    Fix floating-point precision issues in variational cooling data filenames.
    
    Args:
        results_dir: directory containing the files
        dry_run: if True, only show what would be renamed without actually renaming
        
    Returns:
        tuple: (files_processed, files_renamed)
    """
    if not os.path.exists(results_dir):
        print(f"Results directory '{results_dir}' does not exist!")
        return 0, 0
    
    files_processed = 0
    files_renamed = 0
    
    print(f"Scanning directory: {results_dir}")
    print(f"Mode: {'DRY RUN' if dry_run else 'ACTUAL RENAMING'}")
    print("=" * 60)
    
    # Get all JSON files in the directory
    all_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
    variational_files = [f for f in all_files if f.startswith('variational_cooling_data_')]
    
    print(f"Found {len(variational_files)} variational cooling data files")
    print()
    
    for filename in variational_files:
        files_processed += 1
        
        # Parse the filename
        params = parse_variational_cooling_filename(filename)
        if not params:
            print(f"SKIP: Could not parse filename: {filename}")
            continue
        
        # Generate clean filename
        clean_filename = generate_clean_filename(params)
        
        # Check if the filename needs fixing
        if filename == clean_filename:
            print(f"OK:   {filename}")
            continue
        
        # Check if clean filename already exists
        old_path = os.path.join(results_dir, filename)
        new_path = os.path.join(results_dir, clean_filename)
        
        if os.path.exists(new_path):
            print(f"SKIP: Clean filename already exists: {clean_filename}")
            print(f"      Original: {filename}")
            continue
        
        # Rename the file
        print(f"RENAME: {filename}")
        print(f"        -> {clean_filename}")
        
        if not dry_run:
            try:
                shutil.move(old_path, new_path)
                files_renamed += 1
                print(f"        ✓ Renamed successfully")
            except Exception as e:
                print(f"        ✗ Error renaming: {e}")
        else:
            files_renamed += 1
            print(f"        ✓ Would rename")
        
        print()
    
    print("=" * 60)
    print(f"Summary:")
    print(f"  Files processed: {files_processed}")
    print(f"  Files {'would be ' if dry_run else ''}renamed: {files_renamed}")
    
    return files_processed, files_renamed


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fix floating-point precision issues in variational cooling data filenames')
    parser.add_argument('--results-dir', default='results', help='Directory containing the files')
    parser.add_argument('--execute', action='store_true', help='Actually rename files (default is dry run)')
    
    args = parser.parse_args()
    
    print("Floating-Point Filename Fixer")
    print("=" * 50)
    
    # Run the fix
    processed, renamed = fix_floating_point_filenames(args.results_dir, dry_run=not args.execute)
    
    if not args.execute and renamed > 0:
        print()
        print("To actually rename the files, run with --execute flag:")
        print(f"  python3 fix_floating_point_filenames.py --results-dir {args.results_dir} --execute")
