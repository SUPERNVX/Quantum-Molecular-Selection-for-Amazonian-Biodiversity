"""
classical_molecular_selection.py

Classical baseline for molecular diversity optimization.
Implements Greedy and Genetic Algorithm approaches.
Handles data persistence (Similarity Matrix) and Qiskit-compatible bitstring conversion.

Author: Nicolas Mendes de Araújo
Date: February 2026
"""

import os
import time
import numpy as np
import pandas as pd
from typing import List, Set, Tuple
from rdkit import Chem
from rdkit.Chem import DataStructs, rdFingerprintGenerator

class MolecularDiversitySelector:
    """
    Classical algorithm for selecting maximally diverse molecular subset.
    Acts as the 'Ground Truth' and 'Warm-Start Generator' for the Quantum Pipeline.
    """
    
    def __init__(self, molecules_df: pd.DataFrame, cache_dir: str = "data/processed"):
        """
        Args:
            molecules_df: DataFrame with columns ['smiles', 'name', 'species']
            cache_dir: Directory to save/load the similarity matrix
        """
        self.molecules_df = molecules_df.reset_index(drop=True)
        self.cache_dir = cache_dir
        self.fingerprints = None
        self.similarity_matrix = None
        self.n_molecules = len(molecules_df)
        
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        
        # 1. Compute Fingerprints
        self._compute_fingerprints()
        
        # 2. Load or Compute Similarity Matrix
        self._load_or_compute_matrix()

    def _compute_fingerprints(self):
        """Compute Morgan fingerprints (Radius=2, 2048 bits)."""
        print(f"[INFO] Computing fingerprints for {self.n_molecules} molecules...")
        
        mfgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fingerprints = []
        valid_indices = []
        
        for idx, row in self.molecules_df.iterrows():
            mol = Chem.MolFromSmiles(row['smiles'])
            if mol:
                fp = mfgen.GetFingerprint(mol)
                fingerprints.append(fp)
                valid_indices.append(idx)
            else:
                print(f"  [WARN] Invalid SMILES at index {idx}")
        
        self.fingerprints = fingerprints
        self.valid_indices = valid_indices
        
        # Update n_molecules to valid count
        self.n_molecules = len(fingerprints)

    def _load_or_compute_matrix(self):
        """Loads similarity matrix from disk or computes it if missing."""
        import hashlib
        
        # Create a unique key based on the SMILES sequence to avoid collisions
        smiles_seq = "".join(self.molecules_df['smiles'].astype(str))
        subset_hash = hashlib.sha256(smiles_seq.encode()).hexdigest()[:12]
        
        matrix_path = os.path.join(self.cache_dir, f"sim_matrix_N{self.n_molecules}_{subset_hash}.npy")
        
        if os.path.exists(matrix_path):
            print(f"[LOAD] Loading cached similarity matrix from {matrix_path}...")
            self.similarity_matrix = np.load(matrix_path)
            
            # Validation check
            if self.similarity_matrix.shape != (self.n_molecules, self.n_molecules):
                print("  [WARN] Matrix shape mismatch! Recomputing...")
                self._compute_matrix_logic(matrix_path)
        else:
            print(f"[WORK] Cached matrix not found for hash {subset_hash}. Computing new matrix (O(N^2))...")
            self._compute_matrix_logic(matrix_path)

    def _compute_matrix_logic(self, save_path: str):
        """Core logic to compute Tanimoto similarity matrix."""
        n = self.n_molecules
        matrix = np.zeros((n, n))
        
        for i in range(n):
            # BulkTanimoto is faster than pairwise loop
            sims = DataStructs.BulkTanimotoSimilarity(self.fingerprints[i], self.fingerprints[i:])
            for k, sim in enumerate(sims):
                j = i + k
                matrix[i][j] = sim
                matrix[j][i] = sim
                
        self.similarity_matrix = matrix
        np.save(save_path, matrix)
        print(f"[SAVE] Matrix saved to {save_path}")

    def indices_to_bitstring(self, selected_indices: List[int]) -> str:
        """
        Converts a list of selected indices to a Little-Endian Bitstring.
        Essential for Qiskit Warm-Start.
        
        Example: N=5, Selected=[0, 2] -> "00101" (Bits 0 and 2 are 1)
        Note: String index 0 corresponds to Qubit N-1 (MSB).
        """
        # Create list of '0's
        bit_list = ['0'] * self.n_molecules
        
        for idx in selected_indices:
            if 0 <= idx < self.n_molecules:
                # Qiskit Little-Endian: Qubit 0 is the LAST character
                # Map index 'i' to string position 'len - 1 - i'
                bit_list[self.n_molecules - 1 - idx] = '1'
                
        return "".join(bit_list)

    def calculate_diversity_score(self, selected_indices: List[int]) -> float:
        """Calculates total diversity (Sum of 1 - Tanimoto) for selected subset."""
        if len(selected_indices) < 2:
            return 0.0
            
        diversity = 0.0
        selected_list = list(selected_indices)
        
        for i in range(len(selected_list)):
            for j in range(i+1, len(selected_list)):
                idx_i = selected_list[i]
                idx_j = selected_list[j]
                diversity += (1.0 - self.similarity_matrix[idx_i][idx_j])
        
        return diversity
    
    # ==========================================
    # Algorithms
    # ==========================================

    def greedy_selection(self, k: int = 20) -> Tuple[List[int], float, float, str]:
        """
        Greedy Algorithm.
        Returns: (Indices, Score, Time, Bitstring)
        """
        if k > self.n_molecules: k = self.n_molecules
        
        print(f"\n[RUN] Running GREEDY Selection (k={k})...")
        start_time = time.time()
        
        selected = set()
        remaining = set(range(self.n_molecules))
        
        # 1. Select initial molecule (most distinct on average)
        avg_sims = np.mean(self.similarity_matrix, axis=1)
        first_idx = np.argmin(avg_sims) # Lowest similarity = Most distinct
        selected.add(first_idx)
        remaining.remove(first_idx)
        
        # 2. Iteratively add
        for _ in range(k - 1):
            best_idx = -1
            best_marginal_diversity = -1.0
            
            for candidate in remaining:
                # Calculate how much diversity this candidate adds
                marginal_diversity = 0
                for s in selected:
                    marginal_diversity += (1.0 - self.similarity_matrix[candidate][s])
                
                if marginal_diversity > best_marginal_diversity:
                    best_marginal_diversity = marginal_diversity
                    best_idx = candidate
            
            if best_idx != -1:
                selected.add(best_idx)
                remaining.remove(best_idx)
        
        final_indices = sorted(list(selected))
        score = self.calculate_diversity_score(final_indices)
        exec_time = time.time() - start_time
        bitstring = self.indices_to_bitstring(final_indices)
        
        print(f"[OK] Greedy Done in {exec_time:.4f}s | Score: {score:.4f}")
        return final_indices, score, exec_time, bitstring

    def genetic_algorithm(self, k: int = 20, pop_size: int = 50, generations: int = 100) -> Tuple[List[int], float, float, str]:
        """
        Genetic Algorithm.
        Returns: (Indices, Score, Time, Bitstring)
        """
        print(f"\n[GA] Running GENETIC ALGORITHM (k={k}, Gen={generations})...")
        start_time = time.time()
        
        # Initialize Population
        population = [np.random.choice(self.n_molecules, k, replace=False) for _ in range(pop_size)]
        
        best_solution = None
        best_score = -1.0
        
        for gen in range(generations):
            scores = [self.calculate_diversity_score(ind) for ind in population]
            
            # Elitism: Keep best
            max_idx = np.argmax(scores)
            if scores[max_idx] > best_score:
                best_score = scores[max_idx]
                best_solution = population[max_idx]
            
            # Simple Selection & Crossover (Tournament)
            new_pop = [best_solution] # Elitism
            while len(new_pop) < pop_size:
                # Tournament
                candidates = np.random.choice(len(population), 4, replace=False)
                parent1 = population[candidates[np.argmax([scores[i] for i in candidates[:2]])]]
                parent2 = population[candidates[2 + np.argmax([scores[i] for i in candidates[2:]])]]
                
                # Crossover (Single Point)
                cut = np.random.randint(1, k)
                child = np.concatenate((parent1[:cut], parent2[cut:]))
                
                # Ensure uniqueness
                child = np.unique(child)
                while len(child) < k:
                    new_val = np.random.randint(0, self.n_molecules)
                    if new_val not in child:
                        child = np.append(child, new_val)
                if len(child) > k:
                    child = child[:k]
                    
                # Mutation (Swap)
                if np.random.rand() < 0.1:
                    idx_mut = np.random.randint(0, k)
                    val_mut = np.random.randint(0, self.n_molecules)
                    if val_mut not in child:
                        child[idx_mut] = val_mut
                        
                new_pop.append(child)
            
            population = new_pop

        final_indices = sorted(list(best_solution))
        exec_time = time.time() - start_time
        bitstring = self.indices_to_bitstring(final_indices)
        
        print(f"[OK] GA Done in {exec_time:.4f}s | Score: {best_score:.4f}")
        return final_indices, best_score, exec_time, bitstring

# ==============================================================================
# Main Execution
# ==============================================================================
if __name__ == "__main__":
    # Path configuration
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_PATH = os.path.join(BASE_DIR, 'data/processed/brnpdb.csv')
    CACHE_DIR = os.path.join(BASE_DIR, 'data/processed')
    RESULTS_DIR = os.path.join(BASE_DIR, 'data/results')
    
    # Load Data
    if not os.path.exists(DATA_PATH):
        print(f"[ERR] Error: Data file not found at {DATA_PATH}")
        # Create dummy for testing if real data missing
        print("[WARN] Creating DUMMY data for testing...")
        df = pd.DataFrame({'smiles': ['C']*20, 'id': range(20)})
    else:
        df = pd.read_csv(DATA_PATH)
        
    # Initialize Selector
    selector = MolecularDiversitySelector(df, cache_dir=CACHE_DIR)
    
    # Define Parameters
    K_SELECT = 5 # Small K for testing, increase for production
    
    # Run Algorithms
    greedy_res = selector.greedy_selection(k=K_SELECT)
    ga_res = selector.genetic_algorithm(k=K_SELECT, generations=50)
    
    # Save Results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_df = pd.DataFrame({
        'method': ['Greedy', 'GeneticAlgorithm'],
        'indices': [greedy_res[0], ga_res[0]],
        'diversity_score': [greedy_res[1], ga_res[1]],
        'time_sec': [greedy_res[2], ga_res[2]],
        'bitstring': [greedy_res[3], ga_res[3]] # Saved for QAOA
    })
    
    output_path = os.path.join(RESULTS_DIR, 'classical_baseline_results.csv')
    results_df.to_csv(output_path, index=False)
    
    print(f"\n[SAVE] Results saved to: {output_path}")
    print(f"[KEY] Use the Greedy Bitstring for QAOA Warm-Start: {greedy_res[3]}")