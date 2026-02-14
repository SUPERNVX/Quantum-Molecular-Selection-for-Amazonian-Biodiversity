
import numpy as np
import pandas as pd
import os
import json
from src.hotstart.hybrid_selector import HybridQuantumSelector

def diagnose_trap(trap_name="trap_N25_K8"):
    trap_path = os.path.join('data/traps', trap_name)
    df = pd.read_csv(os.path.join(trap_path, 'molecules.csv'))
    with open(os.path.join(trap_path, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    n = metadata['n']
    k = metadata['k']
    
    selector = HybridQuantumSelector(df)
    H_op = selector.formulate_problem(k)
    
    print(f"\n--- Diagnostic for {trap_name} ---")
    print(f"Number of Pauli terms: {len(H_op)}")
    
    # Calculate Energy for Greedy indices from metadata
    greedy_indices = metadata['greedy_indices']
    
    def get_energy(indices):
        # Bitstring vector
        x = np.zeros(n)
        for idx in indices:
            x[idx] = 1
        
        # Binary to Ising: z = 1 - 2x
        z = 1 - 2*x
        
        # Energy = Sum h_i z_i + Sum J_ij z_i z_j
        energy = 0
        # Parse PauliOp
        for pauli, coeff in zip(H_op.paulis, H_op.coeffs):
            label = pauli.to_label()
            indices_active = [i for i, char in enumerate(reversed(label)) if char == 'Z']
            
            p_val = 1.0
            for ia in indices_active:
                p_val *= z[ia]
            
            energy += coeff.real * p_val
            
        return energy

    e_greedy = get_energy(greedy_indices)
    print(f"Energy of Greedy (metadata): {e_greedy:.4f}")
    
    # Test a neighbor (remove one, add one)
    import random
    neighbor_indices = list(greedy_indices)
    removed = neighbor_indices.pop(random.randint(0, k-1))
    available = [i for i in range(n) if i not in greedy_indices]
    added = random.choice(available)
    neighbor_indices.append(added)
    
    e_neighbor = get_energy(neighbor_indices)
    print(f"Energy of Neighbor (k=8):  {e_neighbor:.4f}")
    
    # Test Invalid (k-1)
    invalid_indices = list(greedy_indices)[:-1]
    e_invalid = get_energy(invalid_indices)
    print(f"Energy of Invalid (k=7):   {e_invalid:.4f}")
    
    print(f"\nGap (Invalid-Greedy): {e_invalid - e_greedy:.4f}")
    if (e_invalid - e_greedy) > 10:
        print("RESULT: Healthy constraint enforcement.")
    else:
        print("WARNING: Constraint penalty might be too weak!")

if __name__ == "__main__":
    diagnose_trap()
