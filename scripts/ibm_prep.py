"""
prepare_for_ibm_quantum.py

Prepare molecular dataset for IBM Quantum execution
Reduces problem to manageable size while preserving quality
"""

import numpy as np
import pandas as pd
from typing import List, Tuple
import json


def prepare_quantum_subset(df: pd.DataFrame, 
                          target_size: int = 25,
                          method: str = 'clustering') -> Tuple[pd.DataFrame, dict]:
    """
    Prepare subset of molecules for quantum computation
    
    Args:
        df: Full molecular dataset
        target_size: Size of subset (≤ 30 for IBM Quantum)
        method: 'clustering', 'diversity_sampling', or 'random'
        
    Returns:
        (subset_df, metadata)
    """
    print(f"\n{'='*60}")
    print(f"PREPARING SUBSET FOR IBM QUANTUM")
    print(f"{'='*60}")
    print(f"Original size: {len(df)} molecules")
    print(f"Target size: {target_size} molecules")
    print(f"Method: {method}")
    
    metadata = {
        'original_size': len(df),
        'target_size': target_size,
        'method': method
    }
    
    if method == 'clustering':
        # Use hierarchical clustering + greedy
        import sys
        import os
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        
        from src.quantum.hierarchical_selector import HierarchicalQAOASelector
        
        selector = HierarchicalQAOASelector(df, use_real_quantum=False)
        
        # Cluster and select representatives
        cluster_labels = selector.hierarchical_clustering(n_clusters=10)
        
        # Select top molecules from each cluster using greedy
        representatives = []
        per_cluster = max(2, target_size // 10)
        
        for cluster_id in range(10):
            cluster_indices = np.where(cluster_labels == cluster_id)[0].tolist()
            
            if len(cluster_indices) > 0:
                cluster_df = df.iloc[cluster_indices]
                
                from src.classical.classical_molecular_selection import MolecularDiversitySelector
                cluster_selector = MolecularDiversitySelector(cluster_df)
                
                k = min(per_cluster, len(cluster_indices))
                selected_local, _, _ = cluster_selector.greedy_selection(k=k)
                
                selected_global = [cluster_indices[i] for i in selected_local]
                representatives.extend(selected_global)
        
        # If we have too many, trim to target_size
        if len(representatives) > target_size:
            representatives = representatives[:target_size]
        
        subset_df = df.iloc[representatives].reset_index(drop=True)
        metadata['representatives'] = representatives
    
    elif method == 'diversity_sampling':
        # MaxMin diversity sampling
        from src.classical.classical_molecular_selection import MolecularDiversitySelector
        
        selector = MolecularDiversitySelector(df)
        selected, _, _ = selector.greedy_selection(k=target_size)
        
        subset_df = df.iloc[list(selected)].reset_index(drop=True)
        metadata['selected_indices'] = list(selected)
    
    elif method == 'random':
        # Simple random sampling
        subset_df = df.sample(n=target_size, random_state=42).reset_index(drop=True)
        metadata['seed'] = 42
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    print(f"\nSubset prepared:")
    print(f"  Final size: {len(subset_df)} molecules")
    print(f"  Coverage: {(len(subset_df)/len(df))*100:.2f}% of original dataset")
    
    return subset_df, metadata


def estimate_ibm_quantum_time(n_qubits: int, p: int = 1, shots: int = 512) -> dict:
    """
    Estimate execution time on IBM Quantum
    
    Args:
        n_qubits: Number of qubits (molecules)
        p: QAOA depth
        shots: Number of measurements
        
    Returns:
        time_estimates: Dictionary with timing breakdown
    """
    # Circuit compilation time (empirical)
    compile_time = 5 + (n_qubits * 0.2)  # seconds
    
    # Queue time (highly variable, use average)
    queue_time_min = 60  # 1 minute minimum
    queue_time_avg = 300  # 5 minutes average
    queue_time_max = 7200  # 2 hours maximum
    
    # Execution time on hardware
    # Approximate: 1μs per gate × depth × shots
    gates_per_layer = n_qubits * (n_qubits - 1) / 2  # ZZ gates
    gates_per_layer += n_qubits  # RZ gates
    gates_per_layer += n_qubits  # RX gates (mixer)
    
    total_gates = gates_per_layer * p
    execution_time_per_shot = total_gates * 1e-6  # 1μs per gate
    execution_time_total = execution_time_per_shot * shots
    
    # Classical optimization loop
    optimization_iterations = 50  # COBYLA iterations
    optimization_time = compile_time + (queue_time_avg + execution_time_total) * optimization_iterations
    
    estimates = {
        'n_qubits': n_qubits,
        'p': p,
        'shots': shots,
        'compile_time_sec': compile_time,
        'queue_time_min_sec': queue_time_min,
        'queue_time_avg_sec': queue_time_avg,
        'queue_time_max_sec': queue_time_max,
        'execution_time_per_job_sec': execution_time_total,
        'optimization_iterations': optimization_iterations,
        'total_time_best_case_min': (compile_time + queue_time_min + execution_time_total * optimization_iterations) / 60,
        'total_time_avg_case_min': (compile_time + queue_time_avg + execution_time_total * optimization_iterations) / 60,
        'total_time_worst_case_min': (compile_time + queue_time_max + execution_time_total * optimization_iterations) / 60,
    }
    
    return estimates


def create_ibm_quantum_plan(df: pd.DataFrame, k: int = 36):
    """
    Create complete execution plan for IBM Quantum
    """
    print(f"\n{'='*60}")
    print(f"IBM QUANTUM EXECUTION PLAN")
    print(f"{'='*60}")
    
    # Test different subset sizes
    subset_sizes = [20, 30, 40, 50]
    
    plans = []
    
    for n in subset_sizes:
        print(f"\n{'='*60}")
        print(f"PLAN FOR N={n} MOLECULES")
        print(f"{'='*60}")
        
        # Prepare subset
        subset_df, metadata = prepare_quantum_subset(df, target_size=n, method='clustering')
        
        # Estimate time
        time_est = estimate_ibm_quantum_time(n_qubits=n, p=1, shots=512)
        
        print(f"\nTime Estimates:")
        print(f"  Best case: {time_est['total_time_best_case_min']:.1f} minutes")
        print(f"  Average case: {time_est['total_time_avg_case_min']:.1f} minutes")
        print(f"  Worst case: {time_est['total_time_worst_case_min']:.1f} minutes")
        
        # Check if feasible with 10-minute budget
        budget_minutes = 10
        
        if time_est['total_time_avg_case_min'] <= budget_minutes:
            feasibility = "[OK] FEASIBLE"
        elif time_est['total_time_best_case_min'] <= budget_minutes:
            feasibility = "[WARN] TIGHT (depends on queue)"
        else:
            feasibility = "[FAIL] EXCEEDS BUDGET"
        
        print(f"\nFeasibility (10-min budget): {feasibility}")
        
        plans.append({
            'n_qubits': n,
            'subset_df': subset_df,
            'metadata': metadata,
            'time_estimates': time_est,
            'feasibility': feasibility
        })
        
        # Save subset for this size
        subset_df.to_csv(f'data/processed/ibm_quantum_subset_n{n}.csv', index=False)
        print(f"  Saved: data/processed/ibm_quantum_subset_n{n}.csv")
    
    # Recommendation
    print(f"\n{'='*60}")
    print(f"RECOMMENDATION")
    print(f"{'='*60}")
    
    feasible_plans = [p for p in plans if 'FEASIBLE' in p['feasibility'] or 'TIGHT' in p['feasibility']]
    
    if feasible_plans:
        best_plan = max(feasible_plans, key=lambda x: x['n_qubits'])  # Largest feasible
        
        print(f"[OK] Recommended configuration:")
        print(f"  N = {best_plan['n_qubits']} molecules")
        print(f"  K = {min(k, best_plan['n_qubits']-1)} selections")
        print(f"  Expected time: {best_plan['time_estimates']['total_time_avg_case_min']:.1f} minutes")
        print(f"  File: data/processed/ibm_quantum_subset_n{best_plan['n_qubits']}.csv")
    else:
        print(f"[WARN] No configuration fits 10-minute budget")
        print(f"  Consider:")
        print(f"  - Reducing shots (512 → 256)")
        print(f"  - Reducing subset size (n < 20)")
        print(f"  - Running during off-peak hours (lower queue time)")
    
    return plans


if __name__ == "__main__":
    # Load full dataset
    print("Loading molecular dataset...")
    df = pd.read_csv('data/processed/brnpdb.csv')
    print(f"Loaded {len(df)} molecules")
    
    # Create execution plan
    plans = create_ibm_quantum_plan(df, k=36)
    
    # Save plan
    plans_serializable = []
    for plan in plans:
        plans_serializable.append({
            'n_qubits': plan['n_qubits'],
            'feasibility': plan['feasibility'],
            'time_estimates': plan['time_estimates']
        })
    
    with open('data/results/ibm_quantum_execution_plan.json', 'w') as f:
        json.dump(plans_serializable, f, indent=2)
    
    print("\nExecution plan saved to: data/results/ibm_quantum_execution_plan.json")
