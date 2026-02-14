
import pandas as pd
import os
import json
import time
from src.hotstart.hybrid_selector import HybridQuantumSelector

def run_trap_demo():
    trap_name = "trap_N25_K8"
    trap_path = os.path.join('data/traps', trap_name)
    df = pd.read_csv(os.path.join(trap_path, 'molecules.csv'))
    with open(os.path.join(trap_path, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
        
    n = metadata['n']
    k = metadata['k']
    
    print(f"--- DEMO: Quantum Refinement on {trap_name} ---")
    print(f"Classical Greedy Score: {metadata['greedy_score']:.6f}")
    print(f"Genetic Algorithm Goal: {metadata['ga_score']:.6f}")
    
    selector = HybridQuantumSelector(df)
    
    # Run Hybrid optimization
    # Use the greedy indices from metadata to ensure the seletor starts from the EXACT trap point
    initial_indices = metadata.get('greedy_indices')
    
    # Recalculate the baseline using the CURRENT metric (New Ruler)
    current_greedy_score = selector.classical_selector.calculate_diversity_score(initial_indices)
    
    start_time = time.time()
    h_indices, h_score, h_time = selector.run_hybrid_optimize(
        k=k, p=2, alpha=0.8, maxiter=100, initial_indices=initial_indices
    )
    elapsed = time.time() - start_time
    
    print(f"\n--- Final Results ---")
    print(f"Classical Baseline (Greedy): {current_greedy_score:.6f}")
    print(f"QAOA Refinement Result:      {h_score:.6f}")
    print(f"Genetic Algorithm Goal:      {metadata['ga_score']:.6f}")
    
    if h_score > current_greedy_score + 1e-4:
        improvement = ((h_score - current_greedy_score) / current_greedy_score) * 100
        print(f"SUCCESS: QAOA improved Greedy by {improvement:.4f}%!")
    else:
        print("QAOA matched Greedy but did not surpass it.")
    
    print(f"Total Execution Time: {elapsed:.2f}s")

if __name__ == "__main__":
    run_trap_demo()
