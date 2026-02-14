
import numpy as np
import pandas as pd
import os
import json
from src.hotstart.lite_selector import QuantumMolecularSelectorLite

def cross_check():
    trap_name = "trap_N15_K6"
    trap_path = os.path.join('data/traps', trap_name)
    df = pd.read_csv(os.path.join(trap_path, 'molecules.csv'))
    with open(os.path.join(trap_path, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
        
    n = metadata['n']
    k = metadata['k']
    greedy_idx = metadata['greedy_indices']
    ga_idx = metadata['ga_indices']
    
    selector = QuantumMolecularSelectorLite(df, cache_dir="data/temp_cross")
    H, offset = selector.formulate_problem(k)
    
    penalty = 5.0
    S = selector.similarity_matrix
    
    def get_qubo_energy(indices):
        # QUBO = lambda(sum x - k)^2 + sum S_ij x_i x_j
        term1 = penalty * (len(indices) - k)**2
        term2 = 0.0
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                term2 += S[indices[i]][indices[j]]
        return term1 + term2

    def get_ising_energy(indices):
        val = 0.0
        for pauli, coeff in zip(H.paulis, H.coeffs):
            label = pauli.to_label()
            term_ev = 1.0
            for i in range(n):
                char = label[n - 1 - i]
                if char == 'Z':
                    x_i = 1 if i in indices else 0
                    term_ev *= (1 - 2 * x_i)
            val += coeff.real * term_ev
        return val + offset

    print(f"--- CROSS CHECK N={n}, K={k} ---")
    print(f"Offset: {offset:.6f}")
    
    # Check a few coefficients
    # h9 and h12 are the ones that differ between Greedy and GA
    # Bit 9 is in Greedy, not GA. Bit 12 is in GA, not Greedy.
    
    pauli_dict = {p.to_label(): c.real for p, c in zip(H.paulis, H.coeffs)}
    
    def get_h(idx):
        label = ['I'] * n
        label[n - 1 - idx] = 'Z'
        return pauli_dict.get("".join(label), 0.0)

    print(f"h_9:  {get_h(9):.6f}")
    print(f"h_12: {get_h(12):.6f}")
    
    g_q = get_qubo_energy(greedy_idx)
    g_i = get_ising_energy(greedy_idx)
    print(f"Greedy | QUBO: {g_q:.6f} | Ising: {g_i:.6f} | Match: {abs(g_q-g_i)<1e-7}")
    
    ga_q = get_qubo_energy(ga_idx)
    ga_i = get_ising_energy(ga_idx)
    print(f"GA     | QUBO: {ga_q:.6f} | Ising: {ga_i:.6f} | Match: {abs(ga_q-ga_i)<1e-7}")

if __name__ == "__main__":
    cross_check()
