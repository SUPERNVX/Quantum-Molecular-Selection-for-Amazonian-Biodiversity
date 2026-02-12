"""
hierarchical_qaoa_selector.py

Hybrid Quantum-Classical approach for large-scale molecular selection
Uses hierarchical clustering to break problem into manageable sub-problems
"""

import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.cluster import AgglomerativeClustering
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, rdMolDescriptors, rdFingerprintGenerator
import time

import os
import sys

# Add project root to path to allow running from anywhere
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.append(root_dir)
# Also add current directory for local module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from quantum_molecular_selection import QuantumMolecularSelector


class HierarchicalQAOASelector:
    """
    Scalable molecular selection using hierarchical QAOA
    
    Strategy:
    1. Cluster N molecules into C clusters (~100 molecules each)
    2. Apply QAOA to each cluster independently (parallel)
    3. Combine top molecules from each cluster
    4. Final QAOA on combined set
    
    Complexity Reduction:
    - Original: O(2^N) with N=1078 → IMPOSSIBLE
    - Hierarchical: C × O(2^100) + O(2^40) → FEASIBLE
    """
    
    def __init__(self, molecules_df: pd.DataFrame, use_real_quantum: bool = False):
        self.molecules_df = molecules_df
        self.use_real_quantum = use_real_quantum
        
        # Pre-compute fingerprints for clustering
        self.fingerprints = self._compute_fingerprints()
        self.similarity_matrix = self._compute_similarity_matrix()
    
    def _compute_fingerprints(self) -> List:
        """Compute Morgan fingerprints"""
        print("Computing molecular fingerprints...")
        fingerprints = []
        
        # Morgan fingerprint, radius=2, 2048 bits
        # Using new generator API
        mfgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        
        for smiles in self.molecules_df['smiles']:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                fp = mfgen.GetFingerprint(mol)
                fingerprints.append(fp)
        
        return fingerprints
    
    def _compute_similarity_matrix(self) -> np.ndarray:
        """Compute pairwise Tanimoto similarities"""
        print("Computing similarity matrix...")
        n = len(self.fingerprints)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                sim = DataStructs.TanimotoSimilarity(
                    self.fingerprints[i],
                    self.fingerprints[j]
                )
                similarity_matrix[i][j] = sim
                similarity_matrix[j][i] = sim
        
        return similarity_matrix
    
    def hierarchical_clustering(self, n_clusters: int = 10) -> np.ndarray:
        """
        Cluster molecules using agglomerative clustering
        
        Args:
            n_clusters: Number of clusters to create
            
        Returns:
            cluster_labels: Array of cluster assignments for each molecule
        """
        print(f"\n{'='*60}")
        print(f"HIERARCHICAL CLUSTERING")
        print(f"{'='*60}")
        print(f"Number of molecules: {len(self.molecules_df)}")
        print(f"Target clusters: {n_clusters}")
        
        # Convert similarity to distance matrix
        distance_matrix = 1.0 - self.similarity_matrix
        
        # Agglomerative clustering
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='precomputed',
            linkage='average'
        )
        
        cluster_labels = clustering.fit_predict(distance_matrix)
        
        # Report cluster sizes
        unique, counts = np.unique(cluster_labels, return_counts=True)
        print(f"\nCluster sizes:")
        for cluster_id, count in zip(unique, counts):
            print(f"  Cluster {cluster_id}: {count} molecules")
        
        return cluster_labels
    
    def select_from_cluster(self, 
                           cluster_indices: List[int], 
                           k: int,
                           method: str = 'qaoa') -> Tuple[List[int], float]:
        """
        Select k molecules from a single cluster
        
        Args:
            cluster_indices: Indices of molecules in this cluster
            k: Number of molecules to select
            method: 'qaoa', 'greedy', or 'ga'
            
        Returns:
            (selected_global_indices, diversity_score)
        """
        # Create subset DataFrame
        cluster_df = self.molecules_df.iloc[cluster_indices].reset_index(drop=True)
        
        if method == 'qaoa':
            # Check cluster size to avoid memory overflow in simulator
            if len(cluster_df) > 20: 
                print(f"  Warning: Cluster too large for QAOA Simulator ({len(cluster_df)} > 20). Switching to Greedy for this cluster.")
                method = 'greedy' # Fallback to greedy
            else:
                # Use QAOA
                selector = QuantumMolecularSelector(
                    cluster_df, 
                    use_real_quantum=self.use_real_quantum
                )
                
                # CRITICAL: Adjust k if cluster is too small
                k_adjusted = max(1, min(k, len(cluster_df)))
                
                selected_local, diversity, _ = selector.qaoa_optimize(
                    k=k_adjusted,
                    p=1,
                    shots=1024
                )
                
                # Convert local indices back to global indices
                selected_global = [cluster_indices[i] for i in selected_local]
                return selected_global, diversity
            
        if method == 'greedy':
            # Use greedy (faster for comparison)
            try:
                from classical.classical_molecular_selection import MolecularDiversitySelector
            except ImportError:
                from src.classical.classical_molecular_selection import MolecularDiversitySelector
            
            selector = MolecularDiversitySelector(cluster_df)
            k_adjusted = max(1, min(k, len(cluster_df)))
            selected_local, diversity, _ = selector.greedy_selection(k=k_adjusted)
            selected_global = [cluster_indices[i] for i in selected_local]
            return selected_global, diversity        
        else:
            raise ValueError(f"Unknown method: {method}")
        
    
    def hierarchical_select(self,
                           k_total: int = 36,
                           n_clusters: int = None,
                           method: str = 'qaoa') -> Tuple[List[int], float, dict]:
        """
        Main hierarchical selection algorithm
        
        Args:
            k_total: Total number of molecules to select
            n_clusters: Number of clusters to create
            method: Selection method ('qaoa', 'greedy', 'ga')
            
        Returns:
            (selected_indices, diversity_score, stats)
        """
        print(f"\n{'='*60}")
        print(f"HIERARCHICAL MOLECULAR SELECTION")
        print(f"{'='*60}")
        print(f"Total molecules: {len(self.molecules_df)}")
        print(f"Target selection: {k_total}")
        print(f"Method: {method.upper()}")
        
        start_time = time.time()
        
        # Step 1: Clustering
        # Ensure clusters are small enough for QAOA simulation (max 15-20 qubits)
        if n_clusters is None:
            max_size_per_cluster = 15
            n_clusters = max(2, len(self.molecules_df) // max_size_per_cluster)
            print(f"Calculated optimal clusters: {n_clusters} (to keep N <= 15 per cluster)")

        cluster_labels = self.hierarchical_clustering(n_clusters=n_clusters)
        
        # Step 2: Select from each cluster
        k_per_cluster = max(2, k_total // n_clusters + 1)  # Over-sample
        
        print(f"\n{'='*60}")
        print(f"PHASE 1: Selection within clusters")
        print(f"{'='*60}")
        print(f"Selecting {k_per_cluster} molecules per cluster...")
        
        cluster_selections = []
        cluster_diversities = []
        
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0].tolist()
            
            print(f"\nCluster {cluster_id} ({len(cluster_indices)} molecules):")
            
            selected, diversity = self.select_from_cluster(
                cluster_indices,
                k=k_per_cluster,
                method=method
            )
            
            cluster_selections.extend(selected)
            cluster_diversities.append(diversity)
            
            print(f"  Selected: {len(selected)} molecules")
            print(f"  Diversity: {diversity:.4f}")
        
        # Step 3: Final selection from combined candidates
        print(f"\n{'='*60}")
        print(f"PHASE 2: Final selection from candidates")
        print(f"{'='*60}")
        print(f"Total candidates: {len(cluster_selections)}")
        print(f"Final selection: {k_total}")
        
        # Create DataFrame of candidates
        candidates_df = self.molecules_df.iloc[cluster_selections].reset_index(drop=True)
        
        # Apply final selection
        if len(cluster_selections) <= 20 and method == 'qaoa':
            # Can use QAOA for final selection
            print(f"Using QAOA for final selection...")
            
            final_selector = QuantumMolecularSelector(
                candidates_df,
                use_real_quantum=self.use_real_quantum
            )
            
            final_selected_local, final_diversity, _ = final_selector.qaoa_optimize(
                k=k_total,
                p=1,
                shots=1024
            )
            
            # Convert back to original indices
            final_selected_global = [cluster_selections[i] for i in final_selected_local]
        
        else:
            # Fallback to Greedy for Phase 2 if too many candidates or method is Greedy
            print(f"Using Greedy for final selection...")
            
            try:
                from classical.classical_molecular_selection import MolecularDiversitySelector
            except ImportError:
                from src.classical.classical_molecular_selection import MolecularDiversitySelector
            
            final_selector = MolecularDiversitySelector(candidates_df)
            final_selected_local, final_diversity, _ = final_selector.greedy_selection(k=k_total)
            
            # Convert back to original indices
            final_selected_global = [cluster_selections[i] for i in final_selected_local]
        
        execution_time = time.time() - start_time
        
        # Calculate final diversity on original dataset
        actual_diversity = self._calculate_diversity(final_selected_global)
        
        print(f"\n{'='*60}")
        print(f"HIERARCHICAL SELECTION COMPLETE")
        print(f"{'='*60}")
        print(f"Selected molecules: {len(final_selected_global)}")
        print(f"Final diversity: {actual_diversity:.4f}")
        print(f"Total execution time: {execution_time:.2f}s")
        
        stats = {
            'n_clusters': n_clusters,
            'k_per_cluster': k_per_cluster,
            'total_candidates': len(cluster_selections),
            'cluster_diversities': cluster_diversities,
            'phase1_time': execution_time * 0.8,  # Approximate
            'phase2_time': execution_time * 0.2,
            'total_time': execution_time
        }
        
        return final_selected_global, actual_diversity, stats
    
    def _calculate_diversity(self, selected_indices: List[int]) -> float:
        """Calculate diversity score on original dataset"""
        diversity = 0.0
        for i in range(len(selected_indices)):
            for j in range(i+1, len(selected_indices)):
                idx_i = selected_indices[i]
                idx_j = selected_indices[j]
                distance = 1.0 - self.similarity_matrix[idx_i][idx_j]
                diversity += distance
        
        return diversity


def benchmark_hierarchical_vs_baseline(df: pd.DataFrame, k: int = 36):
    """
    Compare hierarchical QAOA against baselines
    """
    print(f"\n{'='*60}")
    print(f"BENCHMARK: Hierarchical QAOA vs Baselines")
    print(f"{'='*60}")
    print(f"Dataset: {len(df)} molecules")
    print(f"Target: {k} selections")
    
    results = {}
    
    # 1. Greedy Baseline (full dataset)
    print(f"\n[1/3] Running Greedy baseline...")
    try:
        from classical.classical_molecular_selection import MolecularDiversitySelector
    except ImportError:
        from src.classical.classical_molecular_selection import MolecularDiversitySelector
    
    greedy_selector = MolecularDiversitySelector(df)
    greedy_selected, greedy_diversity, greedy_time = greedy_selector.greedy_selection(k=k)
    
    results['greedy'] = {
        'selected': greedy_selected,
        'diversity': greedy_diversity,
        'time': greedy_time
    }
    
    print(f"  Diversity: {greedy_diversity:.4f}")
    print(f"  Time: {greedy_time:.2f}s")
    
    # 2. Genetic Algorithm Baseline
    print(f"\n[2/3] Running Genetic Algorithm baseline...")
    ga_selected, ga_diversity, ga_time = greedy_selector.genetic_algorithm(
        k=k,
        population_size=50,
        generations=100
    )
    
    results['ga'] = {
        'selected': ga_selected,
        'diversity': ga_diversity,
        'time': ga_time
    }
    
    print(f"  Diversity: {ga_diversity:.4f}")
    print(f"  Time: {ga_time:.2f}s")
    
    # 3. Hierarchical QAOA
    print(f"\n[3/3] Running Hierarchical QAOA...")
    
    hierarchical_selector = HierarchicalQAOASelector(df, use_real_quantum=False)
    
    # Test different clustering strategies
    n_clusters_options = [5, 10, 15]
    
    for n_clusters in n_clusters_options:
        print(f"\n  Testing with {n_clusters} clusters...")
        
        h_selected, h_diversity, h_stats = hierarchical_selector.hierarchical_select(
            k_total=k,
            n_clusters=n_clusters,
            method='qaoa'
        )
        
        results[f'hierarchical_qaoa_{n_clusters}'] = {
            'selected': h_selected,
            'diversity': h_diversity,
            'time': h_stats['total_time'],
            'stats': h_stats
        }
        
        print(f"    Diversity: {h_diversity:.4f}")
        print(f"    Time: {h_stats['total_time']:.2f}s")
        print(f"    Efficiency vs Greedy: {(h_diversity/greedy_diversity)*100:.2f}%")
    
    # Summary comparison
    print(f"\n{'='*60}")
    print(f"BENCHMARK SUMMARY")
    print(f"{'='*60}")
    
    print(f"\n{'Method':<30} {'Diversity':<12} {'Time (s)':<10} {'Efficiency':<12}")
    print(f"{'-'*70}")
    
    for method, data in results.items():
        efficiency = (data['diversity'] / greedy_diversity) * 100
        print(f"{method:<30} {data['diversity']:<12.4f} {data['time']:<10.2f} {efficiency:<12.2f}%")
    
    return results


if __name__ == "__main__":
    # Load dataset
    print("Loading molecular dataset...")
    df = pd.read_csv('data/processed/brnpdb.csv')
    print(f"Loaded {len(df)} molecules")
    
    # Run benchmark
    results = benchmark_hierarchical_vs_baseline(df, k=36)
    
    # Save results
    import json
    with open('data/results/hierarchical_benchmark.json', 'w') as f:
        # Convert sets to lists for JSON serialization
        results_serializable = {}
        for method, data in results.items():
            results_serializable[method] = {
                'selected': [int(x) for x in data['selected']],
                'diversity': float(data['diversity']),
                'time': float(data['time'])
            }
        json.dump(results_serializable, f, indent=2)
    
    print("\nResults saved to data/results/hierarchical_benchmark.json")
