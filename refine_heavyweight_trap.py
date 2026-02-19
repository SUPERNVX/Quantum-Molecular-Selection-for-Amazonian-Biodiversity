
import pandas as pd
import numpy as np
import os
import json
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
from src.hotstart.classical import MolecularDiversitySelector
from src.hotstart.find_greedy_traps import save_trap, load_existing_best_gap, run_single_trial

def refine_heavyweight_parallel(n=15, k=6, trials=20000, ga_gen=300, n_workers=11):
    print(f"\n{'='*60}")
    print(f"PARALLEL TRAP REFINEMENT: N={n}, K={k}")
    print(f"Trials: {trials} | GA Generations: {ga_gen} | Workers: {n_workers}")
    print(f"{'='*60}")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data/processed/brnpdb.csv')
    temp_cache_dir = os.path.join(base_dir, 'data/temp_traps')
    
    if not os.path.exists(data_path):
        print(f"Error: Data not found at {data_path}")
        return
        
    df_full = pd.read_csv(data_path)
    os.makedirs(temp_cache_dir, exist_ok=True)
    
    # Load current best
    best_gap_percent = load_existing_best_gap(n, k)
    print(f"Current best gap: {best_gap_percent:.6f}%")
    
    # Prepare trial arguments
    print(f"Generating {trials} random samples...")
    trial_args = []
    for i in range(trials):
        df_sample = df_full.sample(n=n).reset_index(drop=True)
        # Reuse the trial runner from find_greedy_traps but with higher GA gen
        trial_args.append((i, df_sample, k, ga_gen, 50, temp_cache_dir))
    
    found_better = False
    
    # Run in parallel
    print(f"Hunting with {n_workers} CPU cores...")
    with Pool(n_workers) as pool:
        # Using list(tqdm(...)) ensures we consume the generator and see the bar
        results = list(tqdm(
            pool.imap(run_single_trial, trial_args, chunksize=10),
            total=trials,
            desc="Refining"
        ))
    
    # Process results
    for result in results:
        if result and result['gap_percent'] > best_gap_percent:
            print(f"\n[VICTORY] New hard trap found!")
            print(f"  Old Gap: {best_gap_percent:.6f}%")
            print(f"  New Gap: {result['gap_percent']:.6f}%")
            
            best_gap_percent = result['gap_percent']
            metadata = {
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
            # Save the better one
            df_trap = pd.DataFrame(result['molecules'])
            save_trap(df_trap, metadata, n, k)
            found_better = True
                
    if not found_better:
        print(f"\nConclusion: No harder trap found in {trials} trials.")
    else:
        print(f"\nConclusion: Trap N={n}, K={k} upgraded successfully.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hunt for harder traps using parallel processing.")
    parser.add_argument('--n', type=int, default=25)
    parser.add_argument('--k', type=int, default=8)
    parser.add_argument('--trials', type=int, default=1000)
    parser.add_argument('--gen', type=int, default=300)
    parser.add_argument('--workers', type=int, default=11)
    args = parser.parse_args()
    
    refine_heavyweight_parallel(
        n=args.n, 
        k=args.k, 
        trials=args.trials, 
        ga_gen=args.gen, 
        n_workers=args.workers
    )
