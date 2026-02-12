"""
vqe_embedding_selector.py

Alternative approach: Reduce problem dimensionality using embeddings
Then apply quantum optimization
"""

import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, rdMolDescriptors, rdFingerprintGenerator


class EmbeddingQAOASelector:
    """
    Dimensionality reduction + QAOA
    
    Strategy:
    1. Embed 1078 molecules into low-dimensional space (e.g., 20D)
    2. Cluster in embedding space
    3. Select representative molecules from each cluster
    4. Apply QAOA to representatives
    
    Advantage: Preserves global structure while reducing problem size
    """
    
    def __init__(self, molecules_df: pd.DataFrame):
        self.molecules_df = molecules_df
        self.fingerprints = self._compute_fingerprints()
    
    def _compute_fingerprints(self) -> np.ndarray:
        """Compute Morgan fingerprints as numpy array"""
        print("Computing molecular fingerprints...")
        fingerprints = []
        
        for smiles in self.molecules_df['smiles']:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                # Morgan fingerprint, radius=2, 2048 bits
                mfgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
                fp = mfgen.GetFingerprint(mol)
                arr = np.zeros((2048,))
                DataStructs.ConvertToNumpyArray(fp, arr)
                fingerprints.append(arr)
        
        return np.array(fingerprints)
    
    def compute_embedding(self, n_components: int = 20, method: str = 'pca') -> np.ndarray:
        """
        Reduce dimensionality using PCA or t-SNE
        
        Args:
            n_components: Target dimensions
            method: 'pca' or 'tsne'
            
        Returns:
            embedded: (N, n_components) array
        """
        print(f"\nComputing {method.upper()} embedding...")
        print(f"  Original dimensions: {self.fingerprints.shape[1]}")
        print(f"  Target dimensions: {n_components}")
        
        if method == 'pca':
            reducer = PCA(n_components=n_components)
            embedded = reducer.fit_transform(self.fingerprints)
            
            explained_var = np.sum(reducer.explained_variance_ratio_)
            print(f"  Explained variance: {explained_var*100:.2f}%")
        
        elif method == 'tsne':
            # t-SNE is slower but better for visualization
            reducer = TSNE(n_components=n_components, random_state=42)
            embedded = reducer.fit_transform(self.fingerprints)
            print(f"  t-SNE embedding complete")
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return embedded
    
    def select_representatives(self, 
                              embedding: np.ndarray, 
                              n_representatives: int = 30) -> List[int]:
        """
        Select representative molecules using k-means++ initialization
        
        This gives well-distributed representatives across chemical space
        
        Args:
            embedding: (N, D) embedded fingerprints
            n_representatives: Number to select
            
        Returns:
            indices: List of representative molecule indices
        """
        from sklearn.cluster import KMeans
        
        print(f"\nSelecting {n_representatives} representatives...")
        
        # Use k-means to find representatives
        kmeans = KMeans(
            n_clusters=n_representatives,
            init='k-means++',
            n_init=10,
            random_state=42
        )
        
        kmeans.fit(embedding)
        
        # Find molecule closest to each cluster center
        representatives = []
        for center in kmeans.cluster_centers_:
            distances = np.linalg.norm(embedding - center, axis=1)
            closest = np.argmin(distances)
            representatives.append(closest)
        
        print(f"  Selected {len(representatives)} representatives")
        
        return representatives
    
    def embedded_selection(self, 
                          k_total: int = 36,
                          n_representatives: int = 30,
                          method: str = 'qaoa') -> Tuple[List[int], float]:
        """
        Main embedding-based selection
        
        Workflow:
        1. Reduce 1078 → 20D embedding
        2. Select 30 representatives
        3. QAOA on 30 → select final k_total
        
        Args:
            k_total: Final number to select
            n_representatives: Intermediate representatives
            method: 'qaoa' or 'greedy'
            
        Returns:
            (selected_indices, diversity_score)
        """
        print(f"\n{'='*60}")
        print(f"EMBEDDING-BASED SELECTION")
        print(f"{'='*60}")
        print(f"Total molecules: {len(self.molecules_df)}")
        print(f"Representatives: {n_representatives}")
        print(f"Final selection: {k_total}")
        
        # Step 1: Compute embedding
        embedding = self.compute_embedding(n_components=20, method='pca')
        
        # Step 2: Select representatives
        representatives = self.select_representatives(embedding, n_representatives)
        
        # Step 3: QAOA on representatives
        print(f"\nApplying {method.upper()} to representatives...")
        
        repr_df = self.molecules_df.iloc[representatives].reset_index(drop=True)
        
        if method == 'qaoa':
            import os
            import sys
            # Add project root to path
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if root_dir not in sys.path:
                sys.path.append(root_dir)
            # Add current directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if current_dir not in sys.path:
                sys.path.append(current_dir)
            
            try:
                from quantum_molecular_selection import QuantumMolecularSelector
            except ImportError:
                from src.quantum.quantum_molecular_selection import QuantumMolecularSelector
            
            selector = QuantumMolecularSelector(repr_df, use_real_quantum=False)
            selected_local, diversity, _ = selector.qaoa_optimize(k=k_total, p=1, shots=1024)
        
        elif method == 'greedy':
            import os
            import sys
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if root_dir not in sys.path:
                sys.path.append(root_dir)
            
            try:
                from classical.classical_molecular_selection import MolecularDiversitySelector
            except ImportError:
                from src.classical.classical_molecular_selection import MolecularDiversitySelector
            
            selector = MolecularDiversitySelector(repr_df)
            selected_local, diversity, _ = selector.greedy_selection(k=k_total)
        
        # Convert back to original indices
        selected_global = [representatives[i] for i in selected_local]
        
        print(f"\nFinal selection:")
        print(f"  Selected: {len(selected_global)} molecules")
        print(f"  Diversity: {diversity:.4f}")
        
        return selected_global, diversity


if __name__ == "__main__":
    # Load dataset
    print("Loading molecular dataset...")
    df = pd.read_csv('data/processed/brnpdb.csv')
    print(f"Loaded {len(df)} molecules")
    
    # Test embedding approach
    selector = EmbeddingQAOASelector(df)
    
    selected, diversity = selector.embedded_selection(
        k_total=36,
        n_representatives=30,
        method='qaoa'
    )
    
    print(f"\nSelected molecules: {selected}")
    print(f"Diversity: {diversity:.4f}")
