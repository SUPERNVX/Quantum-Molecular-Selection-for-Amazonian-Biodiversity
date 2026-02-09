"""
classical_molecular_selection.py

Classical baseline for molecular diversity optimization
Implements greedy and genetic algorithm approaches
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from typing import List, Set, Tuple
import time

class MolecularDiversitySelector:
    """
    Classical algorithm for selecting maximally diverse molecular subset
    
    Problem Definition:
    - Input: Set of N molecules (N=500)
    - Budget: Select K molecules (K=20)
    - Objective: Maximize structural diversity
    - Constraint: Budget limit
    
    Diversity Metric:
    Sum of pairwise Tanimoto distances between selected molecules
    Higher score = more diverse selection
    """
    
    def __init__(self, molecules_df: pd.DataFrame):
        """
        Args:
            molecules_df: DataFrame with columns ['smiles', 'name', 'species']
        """
        self.molecules_df = molecules_df
        self.fingerprints = None
        self.similarity_matrix = None
        
        # Pre-compute molecular fingerprints
        self._compute_fingerprints()
        
    def _compute_fingerprints(self):
        """Compute Morgan fingerprints for all molecules"""
        print("Computing molecular fingerprints...")
        
        fingerprints = []
        valid_indices = []
        
        for idx, row in self.molecules_df.iterrows():
            mol = Chem.MolFromSmiles(row['smiles'])
            if mol:
                # Morgan fingerprint, radius=2, 2048 bits
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                fingerprints.append(fp)
                valid_indices.append(idx)
        
        self.fingerprints = fingerprints
        self.valid_indices = valid_indices
        print(f"Computed {len(fingerprints)} valid fingerprints")
        
    def _compute_similarity_matrix(self):
        """Pre-compute pairwise Tanimoto similarities"""
        print("Computing similarity matrix...")
        
        n = len(self.fingerprints)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                similarity = DataStructs.TanimotoSimilarity(
                    self.fingerprints[i],
                    self.fingerprints[j]
                )
                similarity_matrix[i][j] = similarity
                similarity_matrix[j][i] = similarity
        
        self.similarity_matrix = similarity_matrix
        print("Similarity matrix computed")
        
    def calculate_diversity_score(self, selected_indices: Set[int]) -> float:
        """
        Calculate diversity score for a selection
        
        Diversity = Sum of pairwise distances (1 - similarity)
        
        Args:
            selected_indices: Set of molecule indices
            
        Returns:
            Diversity score (higher is better)
        """
        if self.similarity_matrix is None:
            self._compute_similarity_matrix()
        
        diversity = 0.0
        selected_list = list(selected_indices)
        
        for i in range(len(selected_list)):
            for j in range(i+1, len(selected_list)):
                idx_i = selected_list[i]
                idx_j = selected_list[j]
                
                # Distance = 1 - similarity
                distance = 1.0 - self.similarity_matrix[idx_i][idx_j]
                diversity += distance
        
        return diversity
    
    def greedy_selection(self, k: int = 20) -> Tuple[Set[int], float, float]:
        """
        Greedy algorithm: Iteratively add molecule that maximizes diversity
        
        Algorithm:
        1. Start with empty selection
        2. Add most central molecule (highest avg distance to all others)
        3. Repeat: Add molecule that maximizes diversity increment
        
        Time Complexity: O(K * N)
        
        Args:
            k: Number of molecules to select
            
        Returns:
            (selected_indices, diversity_score, execution_time)
        """
        print(f"\n{'='*60}")
        print(f"GREEDY SELECTION (k={k})")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        if self.similarity_matrix is None:
            self._compute_similarity_matrix()
        
        n = len(self.fingerprints)
        selected = set()
        remaining = set(range(n))
        
        # Step 1: Select most "central" molecule (highest avg distance)
        avg_distances = []
        for i in range(n):
            avg_dist = np.mean([1.0 - self.similarity_matrix[i][j] 
                               for j in range(n) if j != i])
            avg_distances.append(avg_dist)
        
        first_idx = np.argmax(avg_distances)
        selected.add(first_idx)
        remaining.remove(first_idx)
        
        print(f"Initial selection: Molecule {first_idx}")
        
        # Step 2: Iteratively add molecule that maximizes diversity
        for step in range(1, k):
            best_idx = None
            best_score_increment = -np.inf
            
            for candidate in remaining:
                # Calculate diversity increment if we add this candidate
                increment = sum(1.0 - self.similarity_matrix[candidate][sel] 
                              for sel in selected)
                
                if increment > best_score_increment:
                    best_score_increment = increment
                    best_idx = candidate
            
            selected.add(best_idx)
            remaining.remove(best_idx)
            
            current_diversity = self.calculate_diversity_score(selected)
            print(f"Step {step}: Added molecule {best_idx}, "
                  f"Diversity = {current_diversity:.4f}")
        
        final_diversity = self.calculate_diversity_score(selected)
        execution_time = time.time() - start_time
        
        print(f"\nGreedy Result:")
        print(f"  Final Diversity: {final_diversity:.4f}")
        print(f"  Execution Time: {execution_time:.4f}s")
        
        return selected, final_diversity, execution_time
    
    def genetic_algorithm(self, 
                         k: int = 20,
                         population_size: int = 50,
                         generations: int = 100,
                         mutation_rate: float = 0.1) -> Tuple[Set[int], float, float]:
        """
        Genetic Algorithm for molecular selection
        
        Algorithm:
        1. Initialize population of random selections
        2. For each generation:
           a. Evaluate fitness (diversity score)
           b. Select parents (tournament selection)
           c. Crossover (combine parent selections)
           d. Mutate (random swaps)
        3. Return best solution
        
        Time Complexity: O(population_size * generations * k)
        
        Args:
            k: Number of molecules to select
            population_size: Size of population
            generations: Number of generations
            mutation_rate: Probability of mutation
            
        Returns:
            (selected_indices, diversity_score, execution_time)
        """
        print(f"\n{'='*60}")
        print(f"GENETIC ALGORITHM (k={k})")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        n = len(self.fingerprints)
        
        # Initialize population
        population = []
        for _ in range(population_size):
            individual = set(np.random.choice(n, size=k, replace=False))
            population.append(individual)
        
        best_solution = None
        best_fitness = -np.inf
        
        for gen in range(generations):
            # Evaluate fitness
            fitness_scores = [self.calculate_diversity_score(ind) 
                            for ind in population]
            
            # Track best
            gen_best_idx = np.argmax(fitness_scores)
            gen_best_fitness = fitness_scores[gen_best_idx]
            
            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_solution = population[gen_best_idx].copy()
            
            if gen % 10 == 0:
                print(f"Generation {gen}: Best Fitness = {best_fitness:.4f}")
            
            # Selection (tournament)
            new_population = []
            for _ in range(population_size):
                # Tournament selection
                tournament = np.random.choice(population_size, size=3, replace=False)
                tournament_fitness = [fitness_scores[i] for i in tournament]
                winner_idx = tournament[np.argmax(tournament_fitness)]
                parent1 = population[winner_idx]
                
                tournament = np.random.choice(population_size, size=3, replace=False)
                tournament_fitness = [fitness_scores[i] for i in tournament]
                winner_idx = tournament[np.argmax(tournament_fitness)]
                parent2 = population[winner_idx]
                
                # Crossover
                child = self._crossover(parent1, parent2, k, n)
                
                # Mutation
                if np.random.random() < mutation_rate:
                    child = self._mutate(child, n)
                
                new_population.append(child)
            
            population = new_population
        
        execution_time = time.time() - start_time
        
        print(f"\nGenetic Algorithm Result:")
        print(f"  Final Diversity: {best_fitness:.4f}")
        print(f"  Execution Time: {execution_time:.4f}s")
        
        return best_solution, best_fitness, execution_time
    
    def _crossover(self, parent1: Set[int], parent2: Set[int], 
                   k: int, n: int) -> Set[int]:
        """Single-point crossover"""
        # Take half from each parent, fill rest randomly
        p1_list = list(parent1)
        p2_list = list(parent2)
        
        child = set(p1_list[:k//2])
        child.update(p2_list[:k//2])
        
        # Fill to size k
        while len(child) < k:
            child.add(np.random.randint(0, n))
        
        # Trim if over
        if len(child) > k:
            child = set(list(child)[:k])
        
        return child
    
    def _mutate(self, individual: Set[int], n: int) -> Set[int]:
        """Mutation: swap one molecule"""
        individual = individual.copy()
        
        # Remove random molecule
        remove_idx = np.random.choice(list(individual))
        individual.remove(remove_idx)
        
        # Add random new molecule
        available = set(range(n)) - individual
        add_idx = np.random.choice(list(available))
        individual.add(add_idx)
        
        return individual


def main():
    """Example usage"""
    # Load dataset
    print("Loading molecular dataset...")
    df = pd.read_csv('data/processed/amazonian_molecules.csv')
    print(f"Loaded {len(df)} molecules")
    
    # Initialize selector
    selector = MolecularDiversitySelector(df)
    
    # Run greedy algorithm
    greedy_selection, greedy_diversity, greedy_time = selector.greedy_selection(k=20)
    
    # Run genetic algorithm
    ga_selection, ga_diversity, ga_time = selector.genetic_algorithm(
        k=20, 
        population_size=50, 
        generations=100
    )
    
    # Comparison
    print(f"\n{'='*60}")
    print("CLASSICAL ALGORITHMS COMPARISON")
    print(f"{'='*60}")
    print(f"Greedy:")
    print(f"  Diversity: {greedy_diversity:.4f}")
    print(f"  Time: {greedy_time:.4f}s")
    print(f"\nGenetic Algorithm:")
    print(f"  Diversity: {ga_diversity:.4f}")
    print(f"  Time: {ga_time:.4f}s")
    print(f"\nGA Improvement: {(ga_diversity/greedy_diversity - 1)*100:.2f}%")
    
    # Save results
    results = {
        'greedy_selection': list(greedy_selection),
        'greedy_diversity': greedy_diversity,
        'greedy_time': greedy_time,
        'ga_selection': list(ga_selection),
        'ga_diversity': ga_diversity,
        'ga_time': ga_time,
    }
    
    pd.DataFrame([results]).to_csv('data/results/classical_baseline.csv', index=False)
    print("\nResults saved to data/results/classical_baseline.csv")


if __name__ == "__main__":
    main()
