"""
benchmark.py

Comprehensive benchmark comparing all molecular selection algorithms.

Algorithms compared:
1. Greedy (Classical baseline)
2. Genetic Algorithm (Classical optimizer)
3. QAOA Lite (Fast quantum simulation) - optional
4. Hybrid (Greedy + QAOA) - optional

Usage:
    python src/hotstart/benchmark.py --trap trap_N25_K5
    python src/hotstart/benchmark.py --trap trap_N25_K5 --quantum
    python src/hotstart/benchmark.py --all
"""

import os
import sys
import json
import time
import argparse
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

# Add paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import algorithms
from src.hotstart.classical import MolecularDiversitySelector


def run_greedy(selector: MolecularDiversitySelector, k: int) -> Dict:
    """Run Greedy algorithm."""
    print("   [Greedy] Starting...")
    start = time.time()
    indices, score, _, bitstring = selector.greedy_selection(k=k)
    exec_time = time.time() - start
    print(f"   [Greedy] Done in {exec_time:.4f}s | Score: {score:.4f}")
    return {
        'algorithm': 'Greedy',
        'score': score,
        'time': exec_time,
        'indices': indices
    }


def run_ga(selector: MolecularDiversitySelector, k: int, generations: int = 100) -> Dict:
    """Run Genetic Algorithm."""
    print(f"   [GA] Starting with {generations} generations...")
    start = time.time()
    indices, score, _, bitstring = selector.genetic_algorithm(k=k, generations=generations)
    exec_time = time.time() - start
    print(f"   [GA] Done in {exec_time:.4f}s | Score: {score:.4f}")
    return {
        'algorithm': 'GeneticAlgorithm',
        'score': score,
        'time': exec_time,
        'indices': indices
    }


def run_qaoa_lite(df: pd.DataFrame, k: int, p: int = 2) -> Dict:
    """Run QAOA Lite (fast simulation)."""
    print(f"   [QAOA_Lite] Starting with p={p} layers...")
    start = time.time()
    try:
        from src.hotstart.lite_selector import QuantumMolecularSelectorLite
        print("   [QAOA_Lite] Initializing selector...")
        selector = QuantumMolecularSelectorLite(df, cache_dir="data/temp_benchmark")
        print("   [QAOA_Lite] Running optimization...")
        indices, score, exec_time_internal = selector.qaoa_optimize(k=k, p=p)
        
        exec_time = time.time() - start
        print(f"   [QAOA_Lite] Done in {exec_time:.4f}s | Score: {score:.4f}")
        return {
            'algorithm': 'QAOA_Lite',
            'score': score,
            'time': exec_time,
            'indices': indices
        }
    except Exception as e:
        import traceback
        print(f"   [QAOA_Lite] ERROR: {e}")
        traceback.print_exc()
        return {
            'algorithm': 'QAOA_Lite',
            'score': -1.0,
            'time': time.time() - start,
            'error': str(e)
        }


def run_hybrid(df: pd.DataFrame, k: int, p: int = 2, alpha: float = 0.85) -> Dict:
    """Run Hybrid (Greedy + QAOA)."""
    print(f"   [Hybrid] Starting with p={p} layers, alpha={alpha}...")
    start = time.time()
    try:
        from src.hotstart.hybrid_selector import HybridQuantumSelector
        print("   [Hybrid] Initializing selector...")
        selector = HybridQuantumSelector(df, cache_dir="data/temp_benchmark")
        print("   [Hybrid] Running optimization...")
        indices, score, exec_time_internal = selector.run_hybrid_optimize(k=k, p=p, alpha=alpha)
        
        exec_time = time.time() - start
        print(f"   [Hybrid] Done in {exec_time:.4f}s | Score: {score:.4f}")
        return {
            'algorithm': 'Hybrid',
            'score': score,
            'time': exec_time,
            'indices': indices
        }
    except Exception as e:
        import traceback
        print(f"   [Hybrid] ERROR: {e}")
        traceback.print_exc()
        return {
            'algorithm': 'Hybrid',
            'score': -1.0,
            'time': time.time() - start,
            'error': str(e)
        }


def run_benchmark_on_trap(trap_name: str, include_quantum: bool = False, verbose: bool = True) -> pd.DataFrame:
    """
    Run complete benchmark on a single trap.
    
    Args:
        trap_name: Name of the trap (e.g., 'trap_N25_K5')
        include_quantum: Include QAOA Lite and Hybrid algorithms
        verbose: Print progress
        
    Returns:
        DataFrame with benchmark results
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    trap_path = os.path.join(base_dir, 'data/traps', trap_name)
    
    # Load trap data
    molecules_path = os.path.join(trap_path, 'molecules.csv')
    metadata_path = os.path.join(trap_path, 'metadata.json')
    
    if not os.path.exists(molecules_path):
        print(f"Error: Trap not found at {trap_path}")
        return None
    
    print(f"\n[LOAD] Loading trap data from {trap_path}...")
    df = pd.read_csv(molecules_path)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    n = metadata['n']
    k = metadata['k']
    
    if verbose:
        print("\n" + "="*70)
        print(f"BENCHMARK: {trap_name}")
        print(f"N={n} molecules, K={k} to select")
        print(f"Original Gap: {metadata['gap_percent']:.2f}%")
        print("="*70)
    
    results = []
    
    # Initialize classical selector (used by multiple algorithms)
    print("\n[INIT] Initializing classical selector...")
    selector = MolecularDiversitySelector(df, cache_dir=os.path.join(base_dir, "data/temp_benchmark"))
    
    # 1. Greedy
    print("\n[1/2] Running Greedy...")
    result = run_greedy(selector, k)
    results.append(result)
    
    # 2. Genetic Algorithm
    print("\n[2/2] Running Genetic Algorithm...")
    result = run_ga(selector, k, generations=100)
    results.append(result)
    
    # 3. QAOA Lite (optional - can be slow)
    if include_quantum:
        print("\n[3/4] Running QAOA Lite...")
        result = run_qaoa_lite(df, k, p=2)
        results.append(result)
        
        # 4. Hybrid
        print("\n[4/4] Running Hybrid...")
        result = run_hybrid(df, k, p=1)
        results.append(result)
    
    # Create results DataFrame
    df_results = pd.DataFrame(results)
    
    # Add metadata
    df_results['trap'] = trap_name
    df_results['n'] = n
    df_results['k'] = k
    df_results['original_gap'] = metadata['gap_percent']
    df_results['ga_original_score'] = metadata['ga_score']
    
    # Calculate improvement over Greedy
    greedy_score = df_results[df_results['algorithm'] == 'Greedy']['score'].values[0]
    df_results['improvement_vs_greedy'] = (df_results['score'] - greedy_score) / greedy_score * 100
    
    if verbose:
        print("\n" + "-"*70)
        print("SUMMARY")
        print("-"*70)
        print(f"{'Algorithm':<20} | {'Score':<10} | {'Time':<10} | {'vs Greedy':<10}")
        print("-"*70)
        for _, row in df_results.iterrows():
            if row['score'] >= 0:
                print(f"{row['algorithm']:<20} | {row['score']:<10.4f} | {row['time']:<10.4f}s | {row['improvement_vs_greedy']:+.2f}%")
            else:
                print(f"{row['algorithm']:<20} | {'ERROR':<10} | {row['time']:<10.4f}s | {'N/A':<10}")
        print("-"*70)
        
        # Determine winner
        valid_results = df_results[df_results['score'] > 0]
        if len(valid_results) > 0:
            winner = valid_results.loc[valid_results['score'].idxmax()]
            print(f"\nWINNER: {winner['algorithm']} with score {winner['score']:.4f}")
            
            if winner['algorithm'] in ['QAOA_Lite', 'Hybrid']:
                print(">> QUANTUM VICTORY!")
            elif winner['algorithm'] == 'GeneticAlgorithm':
                print(">> CLASSICAL GA VICTORY!")
            else:
                print(">> GREEDY WINS (baseline)")
    
    return df_results


def run_all_traps(include_quantum: bool = False):
    """Run benchmark on all available traps."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    traps_dir = os.path.join(base_dir, 'data/traps')
    
    # Find all traps
    traps = []
    for item in os.listdir(traps_dir):
        if item.startswith('trap_') and os.path.isdir(os.path.join(traps_dir, item)):
            traps.append(item)
    
    if not traps:
        print("No traps found!")
        return
    
    print(f"Found {len(traps)} traps: {traps}")
    
    all_results = []
    for trap in traps:
        df_result = run_benchmark_on_trap(trap, include_quantum=include_quantum, verbose=True)
        if df_result is not None:
            all_results.append(df_result)
    
    # Combine all results
    if all_results:
        df_all = pd.concat(all_results, ignore_index=True)
        
        # Save results
        output_path = os.path.join(base_dir, 'data/results/benchmark_results.csv')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_all.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
        
        # Summary by algorithm
        print("\n" + "="*70)
        print("OVERALL SUMMARY BY ALGORITHM")
        print("="*70)
        summary = df_all.groupby('algorithm').agg({
            'score': 'mean',
            'time': 'mean',
            'improvement_vs_greedy': 'mean'
        }).round(4)
        print(summary)
        
        return df_all
    
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark molecular selection algorithms')
    parser.add_argument('--trap', type=str, help='Trap name (e.g., trap_N25_K5)')
    parser.add_argument('--all', action='store_true', help='Run on all available traps')
    parser.add_argument('--quantum', action='store_true', help='Include quantum algorithms (QAOA Lite, Hybrid)')
    
    args = parser.parse_args()
    
    if args.all:
        run_all_traps(include_quantum=args.quantum)
    elif args.trap:
        run_benchmark_on_trap(args.trap, include_quantum=args.quantum)
    else:
        print("Usage: python benchmark.py --trap trap_N25_K5")
        print("       python benchmark.py --trap trap_N25_K5 --quantum")
        print("       python benchmark.py --all")
