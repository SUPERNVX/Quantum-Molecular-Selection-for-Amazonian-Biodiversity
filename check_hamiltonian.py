
import numpy as np
import pandas as pd
import os
import json
from src.hotstart.lite_selector import QuantumMolecularSelectorLite

def analyze_trap():
    trap_name = "trap_N15_K6"
    base_dir = "."
    trap_path = os.path.join(base_dir, 'data/traps', trap_name)
    molecules_path = os.path.join(trap_path, 'molecules.csv')
    metadata_path = os.path.join(trap_path, 'metadata.json')
    
    df = pd.read_csv(molecules_path)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        
    n = metadata['n']
    k = metadata['k']
    greedy_indices = metadata['greedy_indices']
    ga_indices = metadata['ga_indices']
    
    selector = QuantumMolecularSelectorLite(df, cache_dir="data/temp_diag")
    H, offset = selector.formulate_problem(k)
    
    def get_energy(indices):
        # Convert indices to bitstring
        bitstring = ['0'] * n
        for idx in indices:
            bitstring[idx] = '1'
        bitstring = "".join(bitstring[::-1]) # Little-endian for Qiskit
        
        # Calculate <psi|H|psi> for a basis state
        # In Ising, Z_i |x> = (1 - 2x_i) |x>
        # So expectation value of SparsePauliOp is direct
        val = 0.0
        for pauli, coeff in zip(H.paulis, H.coeffs):
            label = pauli.to_label()
            # Expectation of Z term on bitstring
            # product of (1 - 2*b_i) for i where label[i] == 'Z'
            term_ev = 1.0
            for i, char in enumerate(reversed(label)):
                if char == 'Z':
                    b = int(bitstring[i])
                    term_ev *= (1 - 2 * b)
            val += coeff.real * term_ev
        return val + offset

    e_greedy = get_energy(greedy_indices)
    e_ga = get_energy(ga_indices)
    
    # Calculate Similarity Sums directly
    def get_sim_sum(indices):
        s = 0.0
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                s += selector.similarity_matrix[indices[i]][indices[j]]
        return s

    s_greedy = get_sim_sum(greedy_indices)
    s_ga = get_sim_sum(ga_indices)

    out = []
    out.append(f"--- Energy Analysis for {trap_name} ---")
    out.append(f"Greedy Indices: {greedy_indices} | Score: {metadata['greedy_score']:.6f} | Sim Sum: {s_greedy:.6f} | Energy: {e_greedy:.6f}")
    out.append(f"GA Indices:     {ga_indices} | Score: {metadata['ga_score']:.6f} | Sim Sum: {s_ga:.6f} | Energy: {e_ga:.6f}")
    
    # Expected Score = k*(k-1)/2 - SimSum
    # For k=5, k*(k-1)/2 = 10
    out.append(f"Expected Greedy Score by matrix: {10.0 - s_greedy:.6f}")
    out.append(f"Expected GA Score by matrix:     {10.0 - s_ga:.6f}")
    
    if e_ga < e_greedy:
        out.append(">> Hamiltonian is CORRECT: GA has lower energy than Greedy.")
    else:
        out.append(">> Hamiltonian is WRONG or Penalty is too high: GA energy is higher than Greedy.")
    
    with open("energy_diag.log", "w") as f:
        f.write("\n".join(out))
    print("Results written to energy_diag.log")

if __name__ == "__main__":
    analyze_trap()
