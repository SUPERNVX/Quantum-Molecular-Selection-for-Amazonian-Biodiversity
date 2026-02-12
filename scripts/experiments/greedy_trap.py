"""
create_hard_greedy_trap.py

Gera um cenário sintético onde o algoritmo Greedy cai em uma armadilha (mínimo local)
e o algoritmo Quântico (QAOA/Híbrido) consegue encontrar o ótimo global.
"""
import numpy as np
import pandas as pd
import sys
import os
import time
from typing import Tuple, List

# Adicionar raiz do projeto ao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.quantum.hybrid_selector import HybridQuantumSelector
from src.quantum.lite_selector import QuantumMolecularSelectorLite
from src.classical.classical_molecular_selection import MolecularDiversitySelector

class TrapSelector(HybridQuantumSelector):
    """Subclasse para injetar a matriz de similaridade 'armadilha'"""
    def __init__(self, n: int, k: int):
        self.n = n
        self.k_target = k
        # Criar df fictício com SMILES válidos (alcanos: metano, etano, propano...)
        df = pd.DataFrame({'smiles': ['C' * (i+1) for i in range(n)]})
        super().__init__(df)
        # Forçar a matriz de armadilha APÓS o init do pai
        self.similarity_matrix = self._generate_trap_matrix()

    def _compute_fingerprints(self):
        # Gerar fingerprints reais para evitar erro no init do pai
        fps = []
        valid_idx = []
        for idx, smiles in enumerate(self.molecules_df['smiles']):
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                fps.append(self.mfgen.GetFingerprint(mol))
                valid_idx.append(idx)
        return fps, valid_idx

    def _compute_similarity_matrix(self):
        # Placeholder, será substituído no __init__
        return np.zeros((self.n, self.n))

    def _generate_trap_matrix(self):
        # N=19, k=4
        # 0, 1, 2: "LOCAL WINNERS" (Bait)
        # 10, 11, 12, 13: "GLOBAL PEARLS"
        
        S = np.ones((self.n, self.n))
        np.fill_diagonal(S, 1.0)
        
        # 1. Bait (0, 1, 2) são perfeitos entre si (sim=0)
        for i in [0, 1, 2]:
            for j in [0, 1, 2]:
                if i != j:
                    S[i, j] = 0.0
                    
        # 2. Pearls (10-13) são quase perfeitos entre si (sim=0.05) - Grupo de 4
        for i in range(10, 14):
            for j in range(10, 14):
                if i != j:
                    S[i, j] = 0.05
                    
        # 3. Trap: Bait é muito parecido com as Pearls (sim=0.85)
        for i in [0, 1, 2]:
            for j in range(10, 14):
                S[i, j] = S[j, i] = 0.85
                
        return S

def run_experiment():
    N = 19
    K = 4
    print(f"============================================================")
    print(f"EXPERIMENTO CIENTIFICO OTIMIZADO: GREEDY TRAP (N={N}, K={K})")
    print(f"============================================================")
    
    selector = TrapSelector(N, K)
    results = []

    # 1. Executar Algoritmo Greedy (Classico)
    print("\n[1] Executando Algoritmo Greedy (Classico)...")
    greedy_selector = MolecularDiversitySelector(selector.molecules_df)
    greedy_selector.similarity_matrix = selector.similarity_matrix
    
    start = time.time()
    g_indices, g_div, _ = greedy_selector.greedy_selection(k=K)
    elapsed_g = time.time() - start
    results.append({"Method": "Pure Greedy", "Diversity": g_div, "Time": elapsed_g})
    
    # 2. Executar PURE QAOA (Lite)
    print("\n[2] Executando PURE QAOA (Lite Statevector)...")
    lite = QuantumMolecularSelectorLite(selector.molecules_df)
    lite.similarity_matrix = selector.similarity_matrix
    
    start = time.time()
    # N=19 é o limite do simulador sem gastar 4TB, p=2 é ideal.
    l_indices, l_div = lite.qaoa_optimize_lite(k=K, p=2) 
    elapsed_l = time.time() - start
    
    # Recalcular div real
    l_div_real = 0
    for idx, i in enumerate(l_indices):
        for j in l_indices[idx+1:]:
            l_div_real += (1.0 - selector.similarity_matrix[i][j])
    results.append({"Method": "Pure QAOA (Lite)", "Diversity": l_div_real, "Time": elapsed_l})

    # 3. Executar HYBRID (Greedy + QAOA)
    print("\n[3] Executando HYBRID SELECTOR (Warm Start)...")
    start = time.time()
    h_indices, _ = selector.run_hybrid_optimize(k=K, p=1)
    elapsed_h = time.time() - start
    
    h_div_real = 0
    for idx, i in enumerate(h_indices):
        for j in h_indices[idx+1:]:
            h_div_real += (1.0 - selector.similarity_matrix[i][j])
    results.append({"Method": "Hybrid (Warm-Start)", "Diversity": h_div_real, "Time": elapsed_h})
    
    # Tabela Final
    print(f"\n{'='*70}")
    print(f"FINAL COMPARISON (N={N}, K={K})")
    print(f"{'='*70}")
    print(f"{'Method':<25} {'Diversity':<15} {'Time (s)':<10}")
    print(f"{'-'*65}")
    for res in results:
        print(f"{res['Method']:<25} {res['Diversity']:<15.4f} {res['Time']:<10.4f}")
    print(f"{'='*70}")

    print("\nANALISE FINAL:")
    best_method = max(results, key=lambda x: x['Diversity'])
    if best_method['Method'] == "Pure QAOA (Lite)":
        print(f"VITORIA QUANTICA! O OAOA puro escapou da trap que o Greedy caiu.")
    elif best_method['Method'] == "Hybrid (Warm-Start)" and best_method['Diversity'] > results[0]['Diversity']:
         print(f"VITORIA HIBRIDA! O refinamento quântico superou o baseline clássico.")
    else:
        print("O Greedy ainda e competitivo ou o QAOA precisa de mais profundidade (p).")
    print("="*60)

if __name__ == "__main__":
    run_experiment()
