#!/usr/bin/env python3
"""
Data collection module for variational cooling MPS simulation.
Separates data collection from analysis for better code organization.
"""

import numpy as np
import time
import json
import os
from typing import List, Tuple, Dict, Any

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_aer import AerSimulator

from ising_ground_state_dmrg import calculate_ising_ground_state


def get_noisy_simulator(single_qubit_gate_noise=0.0003, two_qubit_gate_noise=0.003,
                        max_bond_dimension=4):
    """Create a noisy quantum simulator with custom noise model."""
    noise_model = NoiseModel()
    single_qubit_error = depolarizing_error(single_qubit_gate_noise, 1)
    noise_model.add_all_qubit_quantum_error(single_qubit_error, ['rz', 'rx'])

    rzz_error = depolarizing_error(two_qubit_gate_noise, 2)
    noise_model.add_all_qubit_quantum_error(rzz_error, ['rzz'])
    
    simulator = AerSimulator(
        noise_model=noise_model, 
        method='matrix_product_state',
        matrix_product_state_max_bond_dimension=max_bond_dimension,
    )
    return simulator



def create_circuit_with_measurements(p, system_qubits, bath_qubits, parameters, num_sweeps=1,
                                   J=0.4, h=0.6, half=True, open_boundary=1,
                                   observables=None, observable_labels=None):
    """
    Create HVA circuit with measurements inserted at specific positions.
    
    Args:
        p: number of HVA layers per sweep
        system_qubits: number of system qubits
        bath_qubits: number of bath qubits
        parameters: [alpha, beta, B_t, g_t] parameter arrays
        num_sweeps: number of cooling sweeps
        J: Ising coupling strength
        h: transverse field strength
        half: if True, one bath site per two system qubits
        open_boundary: boundary conditions (1=open, 0=periodic)
        observables: list of (pauli_string, coefficient) tuples
        observable_labels: list of labels for observables
    
    Returns:
        QuantumCircuit: circuit with measurements inserted
    """
    alpha, beta, B_t, g_t = parameters
    num_qubits = system_qubits + bath_qubits
    
    qc = QuantumCircuit(system_qubits + bath_qubits)
    
    # Add measurements for initial state (sweep_0)
    if observables and observable_labels:
        for i, (pauli_string, coefficient) in enumerate(observables):
            # Create SparsePauliOp for this observable
            observable_op = SparsePauliOp.from_list([(pauli_string, coefficient)])
            qc.save_expectation_value(
                observable_op, 
                list(range(observable_op.num_qubits)), 
                label=f"sweep_0_{observable_labels[i]}", 
                pershot=True
            )
    
    for sweep in range(num_sweeps):    
        for i in range(p):   
            # H_1: brick-wall layout of RZZ gates on system bonds, and Z-rotations on bath qubits
            for j in range(0, system_qubits-open_boundary, 2):
                qc.rzz(-J*alpha[i], j, (j+1)%system_qubits)
            for j in range(1, system_qubits-open_boundary, 2):
                qc.rzz(-J*alpha[i], j, (j+1)%system_qubits)                

            for j in range(system_qubits, num_qubits):
                qc.rz(-B_t[i], j)
                
            # H_2: RX rotations on system qubits
            for j in range(system_qubits):
                qc.rx(-h*beta[i], j)
                
            # H_3: RYY gates connecting system and bath qubits
            if half:
                for j in range(bath_qubits):
                    qc.ryy(-g_t[i], 2*j, j+system_qubits)
            else:
                for j in range(bath_qubits):
                    qc.ryy(-g_t[i], j, j+system_qubits)
        
        # Add measurements after this sweep (before reset) - this becomes sweep_{sweep+1}
        if observables and observable_labels:
            for i, (pauli_string, coefficient) in enumerate(observables):
                # Create SparsePauliOp for this observable
                observable_op = SparsePauliOp.from_list([(pauli_string, coefficient)])
                qc.save_expectation_value(
                    observable_op, 
                    list(range(observable_op.num_qubits)), 
                    label=f"sweep_{sweep+1}_{observable_labels[i]}", 
                    pershot=True
                )
        
        # Reset bath qubits after each sweep
        for k in range(bath_qubits):
            qc.reset(-k-1)
    
    return qc


def get_best_parameters(J, h, p):
    """Get pre-optimized parameters for specific J, h, p combinations."""
    if J == 0.4 and h == 0.6 and p == 3:
        best_para = np.array([0.026932368894753006, 0.58775691792609, 1.416345878671235, -0.5847230425335908, 2.108254043912492, 0.9685586146293224, 3.141592653589793, 1.9041272021498745, -0.7655691857212363, 1.463995846549075, 1.062269199105236, 0.450403882505529])
    elif J == 0.45 and h == 0.55 and p == 3:
        best_para = np.array([0.2579068792545095, 0.6394665483430537, 1.6000809859220193, -0.4677029550794296, 2.251594733293615, 1.0956802728896018, 3.141592653589793, 1.92078829520187, -0.8305840116887662, 1.2839351722098988, 0.9409858928175381, 0.4159040499271145])
    elif J == 0.55 and h == 0.45 and p == 3:
        best_para = np.array([0.13400875081625385, 0.5957504609239808, 1.5744245798463257, -0.6538439040734678, 2.449280029884097, 1.1426198430174903, 3.141592653589793, 2.039226478246942, -0.687046687337217, 1.1995712980171938, 0.9917912365595293, 0.2874759768113605])
    elif J == 0.6 and h == 0.4 and p == 3:
        best_para = np.array([0.2758651574486719, 0.39292632015710316, 1.559163323925934, -1.6112927238917285, 3.141592653589793, 1.1794018466868508, 3.141592653589793, 2.0377323161522973, -0.8765720657695009, 1.196657733890989, 0.7242895048133942, 0.39439092939668874])
    else:
        raise ValueError(f"No pre-optimized parameters found for J={J}, h={h}, p={p}")
    return best_para


def create_system_observables(system_qubits, bath_qubits):
    """
    Create all system observables: X_i, Z_i, and Z_i Z_j for all system qubits.
    
    Args:
        system_qubits: number of system qubits
        bath_qubits: number of bath qubits
    
    Returns:
        tuple: (observables, observable_labels)
    """
    observables = []
    observable_labels = []
    
    # Single-qubit X observables for all system qubits
    for i in range(system_qubits):
        pauli_str = "I" * bath_qubits
        for k in range(system_qubits):
            if k == i:
                pauli_str += "X"
            else:
                pauli_str += "I"
        observables.append((pauli_str, 1.0))
        observable_labels.append(f"X_{i}")
    
    # Single-qubit Z observables for all system qubits
    for i in range(system_qubits):
        pauli_str = "I" * bath_qubits
        for k in range(system_qubits):
            if k == i:
                pauli_str += "Z"
            else:
                pauli_str += "I"
        observables.append((pauli_str, 1.0))
        observable_labels.append(f"Z_{i}")
    
    # Two-qubit ZZ observables for all pairs of system qubits
    for i in range(system_qubits):
        for j in range(i+1, system_qubits):  # Only upper triangular to avoid duplicates
            pauli_str = "I" * bath_qubits
            for k in range(system_qubits):
                if k == i or k == j:
                    pauli_str += "Z"
                else:
                    pauli_str += "I"
            observables.append((pauli_str, 1.0))
            observable_labels.append(f"ZZ_{i}_{j}")
    
    return observables, observable_labels


def collect_variational_cooling_data(system_qubits: int, bath_qubits: int, open_boundary: int,
                                   J: float, h: float, p: int, num_sweeps: int,
                                   single_qubit_gate_noise: float, two_qubit_gate_noise: float,
                                   training_method: str, initial_state: str,
                                   bond_dimensions: List[int], num_shots: int,
                                   output_dir: str = "results") -> Dict[str, Any]:
    """
    Collect data from variational cooling simulation.
    
    Args:
        system_qubits: number of system qubits
        bath_qubits: number of bath qubits
        open_boundary: boundary conditions (1=open, 0=periodic)
        J: Ising coupling strength
        h: transverse field strength
        p: number of HVA layers per sweep
        num_sweeps: number of cooling sweeps
        single_qubit_gate_noise: single qubit gate noise parameter
        two_qubit_gate_noise: two qubit gate noise parameter
        training_method: training method ("energy" or "mpo_fidelity")
        initial_state: initial state description
        bond_dimensions: list of bond dimensions to test
        num_shots: number of shots for each measurement
        output_dir: directory to save results
    
    Returns:
        dict: collected data and metadata
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get best parameters
    best_para = get_best_parameters(J, h, p)
    
    # Split parameters
    lengths = [p, p, p, p]
    split_pts = np.cumsum(lengths)[:-1]
    alpha, beta, B_t, g_t = np.split(best_para, split_pts)
    
    # Create system observables
    observables, observable_labels = create_system_observables(system_qubits, bath_qubits)
    
    # Build circuit with measurements inserted at the right positions
    circuit_with_saves = create_circuit_with_measurements(
        p, system_qubits, bath_qubits, 
        [alpha, beta, B_t, g_t],
        num_sweeps=num_sweeps, J=J, h=h, 
        half=True, open_boundary=open_boundary,
        observables=observables,
        observable_labels=observable_labels
    )
    
    # Setup layout
    indices = list(range(system_qubits + bath_qubits))
    bath_indices = [k for k in range(1, system_qubits + bath_qubits, 3)]
    for bath_idx in bath_indices:
        indices.remove(bath_idx)
    layout = indices + bath_indices
    
    # Results storage - collect raw data from each bond dimension
    raw_results = {}
    
    # Run simulation for each bond dimension
    for bond_dim in bond_dimensions:
        print(f"Testing bond_dim={bond_dim}")
        
        # Setup backend with current bond dimension
        backend = get_noisy_simulator(
            single_qubit_gate_noise=single_qubit_gate_noise,
            two_qubit_gate_noise=two_qubit_gate_noise,
            max_bond_dimension=bond_dim,
        )
        
        # Transpile circuit with layout
        pm = generate_preset_pass_manager(optimization_level=3, backend=backend, initial_layout=layout)
        transpiled_circuit = pm.run(circuit_with_saves)
        
        # Run simulation
        job = backend.run(transpiled_circuit, shots=num_shots)
        result = job.result()
        
        # Extract expectation values and standard deviations
        bond_dim_results = {
            'bond_dim': bond_dim,
            'shots': num_shots,
            'measurements': {}
        }
        
        # Process all measurements (sweep_0 through sweep_{num_sweeps})
        for sweep in range(num_sweeps + 1):  # 0 for initial state, 1 to num_sweeps for after each sweep
            sweep_key = f"sweep_{sweep}"
            bond_dim_results['measurements'][sweep_key] = {}
            
            for i, label in enumerate(observable_labels):
                sweep_label = f"sweep_{sweep}_{label}"
                obs_data = result.data()[sweep_label]
                
                if isinstance(obs_data, np.ndarray) and len(obs_data) > 1:
                    shot_values = obs_data
                    mean_val = float(np.mean(shot_values))
                    variance = np.var(shot_values, ddof=1)
                    std_error = float(np.sqrt(variance / num_shots))
                else:
                    mean_val = float(obs_data)
                    std_error = 0.0
                
                bond_dim_results['measurements'][sweep_key][label] = {
                    'value': mean_val,
                    'std_error': std_error
                }
        
        raw_results[f"bond_dim_{bond_dim}"] = bond_dim_results
    
    # Now compute the final results using the highest bond dimension as reference
    print("Computing final results from multiple bond dimensions...")
    
    # Get bond dimensions sorted
    sorted_bond_dims = sorted(bond_dimensions)
    bd_low = sorted_bond_dims[0]
    bd_high = sorted_bond_dims[-1]
    
    # Create final results structure
    final_results = {
        'shots': num_shots,
        'bond_dimensions_used': bond_dimensions,
        'reference_bond_dim': bd_high,
        'measurements': {}
    }
    
    # Process all measurements (sweep_0 through sweep_{num_sweeps}) with error metrics
    for sweep in range(num_sweeps + 1):  # 0 for initial state, 1 to num_sweeps for after each sweep
        sweep_key = f"sweep_{sweep}"
        final_results['measurements'][sweep_key] = {}
        
        for label in observable_labels:
            # Get values from both bond dimensions
            low_bd_data = raw_results[f"bond_dim_{bd_low}"]['measurements'][sweep_key][label]
            high_bd_data = raw_results[f"bond_dim_{bd_high}"]['measurements'][sweep_key][label]
            
            # Compute error metrics
            mean_val = high_bd_data['value']  # Use high bond dimension value
            truncation_error = float(abs(high_bd_data['value'] - low_bd_data['value']))
            std_error = high_bd_data['std_error']  # Already divided by sqrt(num_shots)
            total_error = float(np.sqrt(truncation_error**2 + std_error**2))
            
            final_results['measurements'][sweep_key][label] = {
                'mean': mean_val,
                'truncation_error': truncation_error,
                'std_error': std_error,
                'total_error': total_error
            }
    
    # Create comprehensive data structure
    collected_data = {
        'metadata': {
            'system_qubits': system_qubits,
            'bath_qubits': bath_qubits,
            'open_boundary': open_boundary,
            'J': J,
            'h': h,
            'p': p,
            'num_sweeps': num_sweeps,
            'single_qubit_gate_noise': single_qubit_gate_noise,
            'two_qubit_gate_noise': two_qubit_gate_noise,
            'training_method': training_method,
            'initial_state': initial_state,
            'bond_dimensions': bond_dimensions,
            'num_shots': num_shots,
            'collection_timestamp': time.time(),
            'observable_labels': observable_labels,
            'error_computation': {
                'description': 'Error metrics computed from multiple bond dimensions',
                'mean': 'Value from highest bond dimension',
                'truncation_error': 'Absolute difference between low and high bond dimension values',
                'std_error': 'Standard error from high bond dimension (std_error/sqrt(num_shots))',
                'total_error': 'Root mean square of truncation_error and std_error'
            }
        },
        'final_results': final_results,
        'raw_data': raw_results  # Keep raw data for debugging/analysis if needed
    }
    
    # Save data to JSON file
    timestamp = int(time.time())
    filename = f"variational_cooling_data_sys{system_qubits}_bath{bath_qubits}_J{J}_h{h}_sweeps{num_sweeps}_noise{single_qubit_gate_noise}_{two_qubit_gate_noise}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(collected_data, f, indent=2, default=str)
    
    print(f"Data saved to: {filepath}")
    
    return collected_data


def compute_ground_state_observables(system_qubits: int, J: float, h: float, 
                                    bond_dim: int = 50, max_iter: int = 200,
                                    output_dir: str = "results") -> Dict[str, Any]:
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
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Calculate ground state using DMRG
        E0, ground_state_mps = calculate_ising_ground_state(
            N=system_qubits, J=J, h=h, bond_dim=bond_dim, max_iter=max_iter, cyclic=False
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
        
        # Save ground state data to JSON file
        timestamp = int(time.time())
        filename = f"ground_state_data_sys{system_qubits}_J{J}_h{h}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(ground_state_data, f, indent=2, default=str)
        
        print(f"Ground state data saved to: {filepath}")
        
        return ground_state_data
        
    except Exception as e:
        print(f"Error computing ground state observables: {e}")
        return {
            'error': str(e),
            'metadata': {
                'system_qubits': system_qubits,
                'J': J,
                'h': h,
                'bond_dim': bond_dim,
                'max_iter': max_iter,
                'computation_timestamp': time.time()
            }
        }


if __name__ == "__main__":
    # Example usage
    print("Data Collection Module for Variational Cooling")
    print("=" * 50)
    
    # Example parameters
    params = {
        'system_qubits': 28,
        'bath_qubits': 14,
        'open_boundary': 1,
        'J': 0.6,
        'h': 0.4,
        'p': 3,
        'num_sweeps': 12,
        'single_qubit_gate_noise': 0.000,
        'two_qubit_gate_noise': 0.00,
        'training_method': 'energy',
        'initial_state': 'zeros',
        'bond_dimensions': [32,64],
        'num_shots': 100
    }
    
    print("Collecting variational cooling data...")
    collected_data = collect_variational_cooling_data(**params)
    
    print("\nComputing ground state observables...")
    ground_state_data = compute_ground_state_observables(
        system_qubits=params['system_qubits'],
        J=params['J'],
        h=params['h']
    )
    
    print("\nData collection completed!")
