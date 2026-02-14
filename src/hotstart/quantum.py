"""
quantum.py

Quantum Molecular Selection with Warm-Start QAOA.
Implements sparse Hamiltonian for large-scale problems.

Features:
- Sparse Hamiltonian with configurable sparsity threshold
- AerSimulator for CPU simulation
- Warm-Start from classical greedy solution

Author: Nicolas Mendes de Araújo
Date: February 2026
"""

import os
import time
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Optional

# Qiskit Core
from qiskit import QuantumCircuit
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# Qiskit Aer (CPU simulator)
try:
    from qiskit_aer import AerSimulator
    AER_AVAILABLE = True
except ImportError:
    AER_AVAILABLE = False
    print("Warning: qiskit-aer not available. Using fake backend.")

# Qiskit Runtime (for IBM Quantum)
from qiskit_ibm_runtime import Session, EstimatorV2 as Estimator, SamplerV2 as Sampler
from qiskit_ibm_runtime.fake_provider import FakeManilaV2

# Local Import
try:
    from .classical import MolecularDiversitySelector
except (ImportError, ValueError):
    import classical
    MolecularDiversitySelector = classical.MolecularDiversitySelector

# ==============================================================================
# 1. Hamiltonian Construction (with Sparsification)
# ==============================================================================

def build_molecular_hamiltonian(
    similarity_matrix, 
    k, 
    penalty_lambda=10.0, 
    sparsity_threshold: Optional[float] = None
):
    """
    Constructs the Hamiltonian QUBO: H = H_cost + H_penalty
    
    H_cost: Minimize similarity (Sum S_ij * x_i * x_j)
    H_penalty: Force constraint of size lambda * (Sum x_i - k)^2
    
    Mapping: x_i = (I - Z_i) / 2
    
    Args:
        similarity_matrix: Pairwise similarity matrix
        k: Number of molecules to select
        penalty_lambda: Constraint penalty strength
        sparsity_threshold: Minimum normalized similarity to include term.
                           If None, uses adaptive threshold based on N.
                           Recommended: 0.1 for N<30, 0.2 for N<80, 0.3 for N>=80
        
    Returns:
        SparsePauliOp: The Hamiltonian operator
    """
    n = len(similarity_matrix)
    
    # Adaptive sparsity threshold
    if sparsity_threshold is None:
        if n < 30:
            sparsity_threshold = 0.1
        elif n < 80:
            sparsity_threshold = 0.2
        else:
            sparsity_threshold = 0.3
    
    # 1. Build Z operators for each qubit
    Z_ops = []
    for i in range(n):
        # Little-endian: "IIZII" where Z is at index i
        op_list = ["I"] * n
        op_list[n - 1 - i] = "Z"  # Qiskit uses inverse order in string
        Z_ops.append(SparsePauliOp("".join(op_list)))
    
    I_op = SparsePauliOp("I" * n)
    
    # 2. Define Number operator (Hamming Weight)
    # N = Sum x_i = Sum (I - Z_i)/2
    Number_op = SparsePauliOp("I" * n, coeffs=[0.0])
    for i in range(n):
        x_i = (I_op - Z_ops[i]) / 2.0
        Number_op += x_i
    
    # 3. Cost Term (Diversity) with Sparsification
    # Minimize Sum_{i<j} S_ij * x_i * x_j
    H_cost = SparsePauliOp("I" * n, coeffs=[0.0])
    
    # Normalize matrix to avoid energy explosion
    max_sim = np.max(similarity_matrix) if np.max(similarity_matrix) > 0 else 1.0
    norm_matrix = similarity_matrix / max_sim

    terms_kept = 0
    terms_total = (n * (n - 1)) // 2

    for i in range(n):
        for j in range(i + 1, n):
            weight = norm_matrix[i, j]
            
            # Only add interaction if similarity is relevant
            # If molecules are very different (sim ~ 0), the term is negligible
            # energetically but computationally expensive (CNOTs). We cut it!
            if weight > sparsity_threshold:
                term = ((I_op - Z_ops[i]) / 2.0) @ ((I_op - Z_ops[j]) / 2.0)
                H_cost += weight * term
                terms_kept += 1

    sparsity_ratio = terms_kept / terms_total if terms_total > 0 else 0
    print(f"Sparsification: Kept {terms_kept}/{terms_total} terms ({sparsity_ratio:.1%})")
    
    # 4. Penalty Term (Constraint)
    # Lambda * (N - k)^2
    diff_op = Number_op - (float(k) * I_op)
    H_penalty = penalty_lambda * (diff_op @ diff_op)
    
    # 5. Total Hamiltonian
    H_total = H_cost + H_penalty
    return H_total.simplify()


def build_biased_initial_state(n_qubits, bitstring, alpha=0.90):
    """
    Creates biased initial state for Warm-Start QAOA.
    
    Args:
        n_qubits: Number of qubits
        bitstring: Classical solution bitstring (Little-Endian)
        alpha: Confidence in classical solution (0.5 = uniform, 0.9 = high confidence)
        
    Returns:
        QuantumCircuit: Initial state circuit
    """
    qc = QuantumCircuit(n_qubits)
    
    # Angles to mix |0> and |1>
    # Ry(theta) -> cos(t/2)|0> + sin(t/2)|1>
    # If bit=1: we want high prob of 1 -> theta ~ pi
    # If bit=0: we want high prob of 0 -> theta ~ 0
    
    # Fine adjustment: don't use exact pi to allow "tunneling"
    theta_idle = 2 * np.arcsin(np.sqrt(1 - alpha))   # Close to 0
    theta_active = 2 * np.arcsin(np.sqrt(alpha))     # Close to pi
    
    # Qiskit bitstring order: char 0 is Qubit N-1
    for i, char in enumerate(bitstring):
        qubit_idx = n_qubits - 1 - i
        if char == '1':
            qc.ry(theta_active, qubit_idx)
        else:
            qc.ry(theta_idle, qubit_idx)
            
    return qc

# ==============================================================================
# 2. Backend Configuration
# ==============================================================================

def get_backend(n_qubits=5, verbose=True):
    """
    Get appropriate backend for CPU simulation.
    
    Args:
        n_qubits: Number of qubits needed
        verbose: Print backend information
        
    Returns:
        Backend instance
    """
    if not AER_AVAILABLE:
        if verbose:
            print("Backend: FakeManilaV2 (5 qubits, for testing only)")
        return FakeManilaV2()
    
    # CPU simulation
    if n_qubits > 30:
        # Use matrix product state for large systems
        backend = AerSimulator(method='matrix_product_state')
        if verbose:
            print("CPU Backend: AerSimulator (Matrix Product State)")
    else:
        backend = AerSimulator(method='statevector')
        if verbose:
            print("CPU Backend: AerSimulator (Statevector)")
    
    return backend

# ==============================================================================
# 3. Main Pipeline
# ==============================================================================

def main():
    """
    Main execution of the Quantum Molecular Selection pipeline.
    """
    print("\n" + "="*60)
    print("QUANTUM MOLECULAR SELECTION (WARM-START QAOA)")
    print("="*60)
    
    # --- Configuration ---
    N_QAOA = 5      # Problem size
    K_SELECT = 2    # Molecules to select
    P_LAYERS = 2    # QAOA layers
    ALPHA = 0.85    # Confidence in Greedy
    PENALTY = 15.0  # Strong penalty for k constraint
    SPARSITY_THRESHOLD = None  # Auto-select based on N
    
    # --- Data ---
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_PATH = os.path.join(BASE_DIR, 'data/processed/brnpdb.csv')
    
    # Load data
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH).head(N_QAOA)
    else:
        print("Warning: Data not found. Using Mock data.")
        df = pd.DataFrame({'smiles': ['C', 'CC', 'CCC', 'CCCC', 'CCCCC'], 'id': range(5)})

    # --- 1. Classical Warm-Start ---
    print(f"\nRunning Greedy Classical (N={N_QAOA}, K={K_SELECT})...")
    selector = MolecularDiversitySelector(df, cache_dir=os.path.join(BASE_DIR, "data/processed/hotstart"))
    greedy_indices, greedy_score, _, greedy_bits = selector.greedy_selection(k=K_SELECT)
    print(f"   >> Bitstring Greedy: {greedy_bits} | Score: {greedy_score:.4f}")

    # --- 2. Quantum Preparation ---
    print("\nBuilding Quantum Circuit...")
    
    # Get backend
    backend = get_backend(n_qubits=N_QAOA)
    
    # Hamiltonian with Sparsity
    H_total = build_molecular_hamiltonian(
        selector.similarity_matrix, 
        K_SELECT, 
        penalty_lambda=PENALTY,
        sparsity_threshold=SPARSITY_THRESHOLD
    )
    
    # Biased Initial State
    initial_state = build_biased_initial_state(N_QAOA, greedy_bits, alpha=ALPHA)
    
    # QAOA Ansatz
    ansatz = QAOAAnsatz(
        cost_operator=H_total, 
        reps=P_LAYERS, 
        initial_state=initial_state,
        name="WarmStart_QAOA"
    )
    
    # ISA Transpilation
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    isa_ansatz = pm.run(ansatz)
    isa_hamiltonian = H_total.apply_layout(isa_ansatz.layout)
    
    # --- 3. Optimization ---
    print(f"\nOptimizing Parameters ({P_LAYERS} layers)...")
    
    objective_history = []
    
    def cost_func(params, estimator, circuit, observables):
        pub = (circuit, observables, params)
        job = estimator.run([pub])
        result = job.result()[0]
        energy = float(result.data.evs)
        objective_history.append(energy)
        return energy

    # Initial parameters (small for Warm-Start)
    x0 = np.random.uniform(-0.1, 0.1, ansatz.num_parameters)
    
    # Use Aer primitives for local simulation
    if AER_AVAILABLE and isinstance(backend, AerSimulator):
        # Local simulation with Aer
        with Session(backend=backend) as session:
            estimator = Estimator(mode=session)
            estimator.options.default_shots = 2048
            
            start_t = time.time()
            res = minimize(
                cost_func,
                x0,
                args=(estimator, isa_ansatz, isa_hamiltonian),
                method='COBYLA',
                options={'maxiter': 60, 'tol': 1e-4}
            )
            end_t = time.time()
            
            print(f"   >> Convergence: {res.message}")
            print(f"   >> Final Energy: {res.fun:.4f}")
            
            # --- 4. Final Sampling ---
            print("\nSampling Results...")
            sampler = Sampler(mode=session)
            sampler.options.default_shots = 2048
            
            final_circuit = isa_ansatz.assign_parameters(res.x)
            final_circuit.measure_all()
            
            job = sampler.run([final_circuit])
            result = job.result()[0]
            counts = result.data.meas.get_counts()
    else:
        # Fallback to fake backend
        with Session(backend=backend) as session:
            estimator = Estimator(mode=session)
            estimator.options.default_shots = 2048
            
            start_t = time.time()
            res = minimize(
                cost_func,
                x0,
                args=(estimator, isa_ansatz, isa_hamiltonian),
                method='COBYLA',
                options={'maxiter': 60, 'tol': 1e-4}
            )
            end_t = time.time()
            
            print(f"   >> Convergence: {res.message}")
            print(f"   >> Final Energy: {res.fun:.4f}")
            
            print("\nSampling Results...")
            sampler = Sampler(mode=session)
            sampler.options.default_shots = 2048
            
            final_circuit = isa_ansatz.assign_parameters(res.x)
            final_circuit.measure_all()
            
            job = sampler.run([final_circuit])
            result = job.result()[0]
            counts = result.data.meas.get_counts()
        
    # --- 5. Result Analysis ---
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    
    best_q_indices = []
    best_q_diversity = -1.0
    
    print("\nTOP QUANTUM SOLUTIONS:")
    print(f"{'Bitstring':<10} | {'Count':<6} | {'Status':<10} | {'Diversity':<10}")
    print("-" * 50)
    
    found_greedy = False
    
    for bitstr, count in sorted_counts[:5]:
        indices = [N_QAOA - 1 - i for i, b in enumerate(bitstr) if b == '1']
        
        is_valid = (len(indices) == K_SELECT)
        status = "OK" if is_valid else f"K!={K_SELECT}"
        
        div_score = 0.0
        if is_valid:
            div_score = selector.calculate_diversity_score(indices)
            
            if div_score > best_q_diversity:
                best_q_diversity = div_score
                best_q_indices = indices
        
        if bitstr == greedy_bits:
            found_greedy = True
            status += " (Greedy)"
            
        print(f"{bitstr:<10} | {count:<6} | {status:<10} | {div_score:.4f}")

    # --- 6. Final Comparison ---
    print("\n" + "="*60)
    print("FINAL VERDICT")
    print("="*60)
    print(f"Greedy Score:  {greedy_score:.6f}")
    print(f"Quantum Score: {best_q_diversity:.6f}")
    
    if abs(best_q_diversity - greedy_score) < 1e-5:
        print(">> TECHNICAL TIE. (QAOA found the global optimum)")
    elif best_q_diversity > greedy_score:
        print(">> QUANTUM VICTORY! (Warm-start refined the solution)")
    else:
        print(">> CLASSICAL VICTORY. (QAOA didn't converge or noise interfered)")

    if not found_greedy:
        print("Note: QAOA lost the Greedy solution. Try increasing Alpha.")


def run_with_trap(trap_path: str, p_layers: int = 3, alpha: float = 0.70, 
                  penalty: float = 15.0):
    """
    Run QAOA on a pre-computed trap dataset.
    
    Args:
        trap_path: Path to trap directory (e.g., 'data/traps/trap_N15_K6')
        p_layers: Number of QAOA layers
        alpha: Warm-start confidence
        penalty: Constraint penalty
    """
    import json
    
    # Load molecules
    molecules_path = os.path.join(trap_path, 'molecules.csv')
    metadata_path = os.path.join(trap_path, 'metadata.json')
    
    if not os.path.exists(molecules_path):
        print(f"Error: Trap not found at {trap_path}")
        return
    
    df = pd.read_csv(molecules_path)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    n = metadata['n']
    k = metadata['k']
    
    print(f"\nRunning QAOA on trap: N={n}, K={k}")
    print(f"Expected gap: {metadata['gap_percent']:.2f}%")
    
    # Initialize selector
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    selector = MolecularDiversitySelector(df, cache_dir=os.path.join(BASE_DIR, "data/temp_traps"))
    
    # Run Greedy
    print(f"\nRunning Greedy...")
    greedy_indices, greedy_score, _, greedy_bits = selector.greedy_selection(k=k)
    print(f"Greedy Score: {greedy_score:.4f}")
    
    # Get backend
    backend = get_backend(n_qubits=n)
    
    # Determine sparsity threshold based on problem size
    # For small simulations (N <= 30), we need 100% of the physics to beat Greedy
    if n <= 30:
        print("[PRECISION] Small simulation detected: Disabling Sparsification for maximum accuracy.")
        sparsity_threshold = -1.0  # Keep 100% of terms
    else:
        print("[SCALE] Large scale detected: Enabling Sparsification.")
        sparsity_threshold = 0.3
    
    # Build Hamiltonian
    print(f"\nBuilding Hamiltonian...")
    H_total = build_molecular_hamiltonian(
        selector.similarity_matrix, 
        k, 
        penalty_lambda=penalty,
        sparsity_threshold=sparsity_threshold
    )
    
    # Build initial state
    initial_state = build_biased_initial_state(n, greedy_bits, alpha=alpha)
    
    # Build ansatz
    ansatz = QAOAAnsatz(
        cost_operator=H_total, 
        reps=p_layers, 
        initial_state=initial_state,
        name="WarmStart_QAOA"
    )
    
    # Transpile
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    isa_ansatz = pm.run(ansatz)
    isa_hamiltonian = H_total.apply_layout(isa_ansatz.layout)
    
    # Optimize
    print(f"\nOptimizing ({p_layers} layers)...")
    
    objective_history = []
    
    def cost_func(params, estimator, circuit, observables):
        pub = (circuit, observables, params)
        job = estimator.run([pub])
        result = job.result()[0]
        energy = float(result.data.evs)
        objective_history.append(energy)
        return energy
    
    # Conservative initialization: start VERY close to zero
    # This ensures the initial circuit is almost identity (preserves Greedy)
    # The optimizer will only move if it finds a real improvement gradient
    x0 = np.random.uniform(-0.01, 0.01, ansatz.num_parameters)
    
    with Session(backend=backend) as session:
        estimator = Estimator(mode=session)
        estimator.options.default_shots = 2048
        
        start_t = time.time()
        res = minimize(
            cost_func,
            x0,
            args=(estimator, isa_ansatz, isa_hamiltonian),
            method='COBYLA',
            options={'maxiter': 100, 'tol': 1e-5, 'rhobeg': 0.05}  # rhobeg controls first step size
        )
        exec_time = time.time() - start_t
        
        print(f"Optimization time: {exec_time:.2f}s")
        print(f"Final energy: {res.fun:.4f}")
        
        # Sample
        print("\nSampling...")
        sampler = Sampler(mode=session)
        sampler.options.default_shots = 4096
        
        final_circuit = isa_ansatz.assign_parameters(res.x)
        final_circuit.measure_all()
        
        job = sampler.run([final_circuit])
        result = job.result()[0]
        counts = result.data.meas.get_counts()
    
    # Analyze
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    
    best_q_diversity = -1.0
    best_q_indices = []
    
    for bitstr, count in sorted_counts[:20]:
        indices = [n - 1 - i for i, b in enumerate(bitstr) if b == '1']
        
        if len(indices) == k:
            div_score = selector.calculate_diversity_score(indices)
            if div_score > best_q_diversity:
                best_q_diversity = div_score
                best_q_indices = indices
    
    # Results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Greedy Score:  {greedy_score:.4f}")
    print(f"QAOA Score:    {best_q_diversity:.4f}")
    print(f"Original GA:   {metadata['ga_score']:.4f}")
    print(f"Original Gap:  {metadata['gap_percent']:.2f}%")
    
    if best_q_diversity > greedy_score:
        improvement = (best_q_diversity - greedy_score) / greedy_score * 100
        print(f"\n>> QAOA improved Greedy by {improvement:.2f}%!")
    else:
        print(f"\n>> QAOA matched Greedy (no improvement)")
    
    return {
        'greedy_score': greedy_score,
        'qaoa_score': best_q_diversity,
        'ga_score': metadata['ga_score'],
        'exec_time': exec_time
    }


if __name__ == "__main__":
    main()
