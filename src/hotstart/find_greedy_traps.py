"""
find_greedy_traps.py

Hunt for "traps" (hard instances) where Genetic Algorithm beats Greedy.
These subsets are ideal for demonstrating quantum advantage with Warm-Start QAOA.

Features:
- Multiprocessing parallelization for faster execution
- Progress tracking with tqdm
- Automatic result saving

Target configurations:
    N=15,  K=6,  Trials=5000
    N=25,  K=5,  Trials=3000
    N=30,  K=5,  Trials=2000
    N=80,  K=12, Trials=200
    N=100, K=11, Trials=150
    N=120, K=10, Trials=100

Author: Nicolas Mendes de Araújo
Date: February 2026
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import sys
import json
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count, Manager
from functools import partial

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import from the same directory (hotstart)
from src.hotstart.classical import MolecularDiversitySelector

# ==============================================================================
# Configuration
# ==============================================================================

# Target configurations: (N, K, Trials)
TARGET_CONFIGS = [
    (15, 6, 5000),
    (25, 8, 10000),
    (30, 5, 2000),
    (80, 12, 200),
    (100, 11, 150),
    (120, 10, 100),
]

# Genetic Algorithm parameters (reduced for speed)
GA_GENERATIONS = 100  # Reduced from 200
GA_POPULATION = 50    # Reduced from 100

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, 'data/processed/brnpdb.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data/traps')
TEMP_CACHE_DIR = os.path.join(BASE_DIR, 'data/temp_traps')

# Number of parallel workers (use all available cores)
N_WORKERS = max(1, cpu_count() - 1)  # Leave one core free

# ==============================================================================
# Worker Function (must be at module level for multiprocessing)
# ==============================================================================

def run_single_trial(args):
    """
    Run a single trial of Greedy vs GA comparison.
    
    Args:
        args: Tuple of (trial_num, df_sample, k, ga_generations, ga_population, temp_cache_dir)
    
    Returns:
        dict with trial results or None if no trap found
    """
    trial_num, df_sample, k, ga_generations, ga_population, temp_cache_dir = args
    
    try:
        # Create selector for this sample
        selector = MolecularDiversitySelector(df_sample, cache_dir=temp_cache_dir)
        
        # Run Greedy
        greedy_idx, greedy_score, _, _ = selector.greedy_selection(k=k)
        
        # Run Genetic Algorithm
        ga_idx, ga_score, _, _ = selector.genetic_algorithm(
            k=k, 
            pop_size=ga_population, 
            generations=ga_generations
        )
        
        # Check if GA beats Greedy (trap found)
        if ga_score > greedy_score + 1e-4:
            gap = ga_score - greedy_score
            gap_percent = (gap / ga_score) * 100
            
            return {
                'trial': trial_num,
                'gap_percent': gap_percent,
                'greedy_score': float(greedy_score),
                'ga_score': float(ga_score),
                'gap': float(gap),
                'greedy_indices': [int(i) for i in greedy_idx],
                'ga_indices': [int(i) for i in ga_idx],
                'molecules': df_sample.to_dict('records')
            }
        
        return None
        
    except Exception as e:
        return None


# ==============================================================================
# Helper Functions
# ==============================================================================

def ensure_directories():
    """Create necessary directories."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEMP_CACHE_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

def save_trap(molecules_df, metadata, n, k):
    """
    Save a trap subset to disk.
    
    Args:
        molecules_df: DataFrame with the molecular subset
        metadata: Dictionary with trap metadata
        n: Number of molecules
        k: Selection size
    """
    # Create trap directory
    trap_dir = os.path.join(OUTPUT_DIR, f'trap_N{n}_K{k}')
    os.makedirs(trap_dir, exist_ok=True)
    
    # Save molecules
    molecules_path = os.path.join(trap_dir, 'molecules.csv')
    molecules_df.to_csv(molecules_path, index=False)
    
    # Save metadata
    metadata_path = os.path.join(trap_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   Saved trap to: {trap_dir}")

def load_existing_best_gap(n, k):
    """
    Load the best gap from existing trap if it exists.
    
    Returns:
        float: Best gap percentage, or -1.0 if no existing trap
    """
    trap_dir = os.path.join(OUTPUT_DIR, f'trap_N{n}_K{k}')
    metadata_path = os.path.join(trap_dir, 'metadata.json')
    
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            return metadata.get('gap_percent', -1.0)
    
    return -1.0

# ==============================================================================
# Main Trap Hunting Function
# ==============================================================================

def hunt_traps():
    """
    Main function to hunt for traps across all target configurations.
    Uses multiprocessing for parallel execution.
    """
    print("\n" + "="*70)
    print("TRAP HUNTING SYSTEM (PARALLEL)")
    print(f"Using {N_WORKERS} parallel workers")
    print("="*70)
    
    # Check data file
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data file not found at {DATA_PATH}")
        return
    
    # Load full dataset
    df_full = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df_full)} molecules from BrNPDB")
    
    # Ensure output directories exist
    ensure_directories()
    
    # Track overall statistics
    overall_stats = {
        'start_time': datetime.now().isoformat(),
        'n_workers': N_WORKERS,
        'ga_generations': GA_GENERATIONS,
        'ga_population': GA_POPULATION,
        'configs': []
    }
    
    # Process each target configuration
    for n, k, trials in TARGET_CONFIGS:
        print(f"\n{'='*70}")
        print(f"CONFIGURATION: N={n}, K={k}, Trials={trials}")
        print(f"{'='*70}")
        
        config_start_time = time.time()
        
        # Load existing best gap
        best_gap = load_existing_best_gap(n, k)
        if best_gap > 0:
            print(f"Existing best gap: {best_gap:.4f}%")
        
        best_trap_df = None
        best_metadata = None
        traps_found = 0
        
        # Generate all trial arguments
        print(f"Generating {trials} random samples...")
        trial_args = []
        for trial in range(trials):
            df_sample = df_full.sample(n=n).reset_index(drop=True)
            trial_args.append((trial, df_sample, k, GA_GENERATIONS, GA_POPULATION, TEMP_CACHE_DIR))
        
        # Run trials in parallel with progress bar
        print(f"Running trials with {N_WORKERS} workers...")
        
        with Pool(N_WORKERS) as pool:
            # Use imap for progress tracking
            results = list(tqdm(
                pool.imap(run_single_trial, trial_args, chunksize=10),
                total=trials,
                desc=f"N={n}, K={k}"
            ))
        
        # Process results
        for result in results:
            if result is not None:
                traps_found += 1
                
                if result['gap_percent'] > best_gap:
                    best_gap = result['gap_percent']
                    best_trap_df = pd.DataFrame(result['molecules'])
                    best_metadata = {
                        'n': n,
                        'k': k,
                        'trial': result['trial'] + 1,
                        'greedy_score': result['greedy_score'],
                        'ga_score': result['ga_score'],
                        'gap': result['gap'],
                        'gap_percent': result['gap_percent'],
                        'greedy_indices': result['greedy_indices'],
                        'ga_indices': result['ga_indices'],
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # Save immediately
                    save_trap(best_trap_df, best_metadata, n, k)
        
        # Configuration summary
        config_time = time.time() - config_start_time
        
        config_stat = {
            'n': n,
            'k': k,
            'trials': trials,
            'traps_found': traps_found,
            'best_gap_percent': float(best_gap) if best_gap > 0 else None,
            'execution_time_sec': config_time,
            'trials_per_second': trials / config_time if config_time > 0 else 0
        }
        overall_stats['configs'].append(config_stat)
        
        # Print summary
        print(f"\n{'='*50}")
        print(f"CONFIGURATION SUMMARY: N={n}, K={k}")
        print(f"{'='*50}")
        print(f"Trials: {trials}")
        print(f"Traps found: {traps_found} ({traps_found/trials*100:.1f}%)")
        if best_gap > 0:
            print(f"Best gap: {best_gap:.4f}%")
            print(f"Saved to: {os.path.join(OUTPUT_DIR, f'trap_N{n}_K{k}')}")
        else:
            print("No traps found. Greedy won all trials.")
        print(f"Execution time: {config_time:.1f}s ({trials/config_time:.1f} trials/sec)")
    
    # Save overall statistics
    overall_stats['end_time'] = datetime.now().isoformat()
    stats_path = os.path.join(OUTPUT_DIR, 'hunting_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(overall_stats, f, indent=2)
    
    # Final summary
    print(f"\n{'='*70}")
    print("HUNTING COMPLETE")
    print(f"{'='*70}")
    print(f"Statistics saved to: {stats_path}")
    print("\nBest traps per configuration:")
    for config in overall_stats['configs']:
        if config['best_gap_percent']:
            print(f"  N={config['n']}, K={config['k']}: {config['best_gap_percent']:.2f}% gap ({config['trials_per_second']:.1f} trials/sec)")
        else:
            print(f"  N={config['n']}, K={config['k']}: No trap found")


def hunt_single_config(n, k, trials, n_workers=None):
    """
    Hunt for traps in a single configuration.
    Useful for testing or running specific configurations.
    
    Args:
        n: Number of molecules
        k: Selection size
        trials: Number of trials
        n_workers: Number of parallel workers (default: auto)
    """
    global N_WORKERS
    if n_workers:
        N_WORKERS = n_workers
    
    # Check data file
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data file not found at {DATA_PATH}")
        return
    
    df_full = pd.read_csv(DATA_PATH)
    ensure_directories()
    
    print(f"\nHunting traps: N={n}, K={k}, Trials={trials}")
    print(f"Workers: {N_WORKERS}")
    
    best_gap = load_existing_best_gap(n, k)
    traps_found = 0
    
    trial_args = []
    for trial in range(trials):
        df_sample = df_full.sample(n=n).reset_index(drop=True)
        trial_args.append((trial, df_sample, k, GA_GENERATIONS, GA_POPULATION, TEMP_CACHE_DIR))
    
    start_time = time.time()
    
    with Pool(N_WORKERS) as pool:
        results = list(tqdm(
            pool.imap(run_single_trial, trial_args, chunksize=10),
            total=trials,
            desc=f"N={n}, K={k}"
        ))
    
    for result in results:
        if result is not None:
            traps_found += 1
            if result['gap_percent'] > best_gap:
                best_gap = result['gap_percent']
                best_trap_df = pd.DataFrame(result['molecules'])
                best_metadata = {
                    'n': n,
                    'k': k,
                    'trial': result['trial'] + 1,
                    'greedy_score': result['greedy_score'],
                    'ga_score': result['ga_score'],
                    'gap': result['gap'],
                    'gap_percent': result['gap_percent'],
                    'greedy_indices': result['greedy_indices'],
                    'ga_indices': result['ga_indices'],
                    'timestamp': datetime.now().isoformat()
                }
                save_trap(best_trap_df, best_metadata, n, k)
    
    exec_time = time.time() - start_time
    print(f"\nDone in {exec_time:.1f}s ({trials/exec_time:.1f} trials/sec)")
    print(f"Traps found: {traps_found} ({traps_found/trials*100:.1f}%)")
    if best_gap > 0:
        print(f"Best gap: {best_gap:.4f}%")


# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Trap Hunting System')
    parser.add_argument('--n', type=int, help='Number of molecules (N)')
    parser.add_argument('--k', type=int, help='Selection size (K)')
    parser.add_argument('--trials', type=int, help='Number of trials')
    parser.add_argument('--workers', type=int, help='Number of parallel workers')
    
    args = parser.parse_args()
    
    if args.n and args.k and args.trials:
        # Run single configuration
        hunt_single_config(args.n, args.k, args.trials, args.workers)
    else:
        # Run all configurations
        hunt_traps()
