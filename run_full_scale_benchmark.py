import pandas as pd
import numpy as np
import time
import os

# Import dos seletores
from classical_molecular_selection import MolecularDiversitySelector as GreedySelector
from quantum_molecular_selection import QuantumMolecularSelector as QAOASelector
from quantum_molecular_selection_optimized import QuantumMolecularSelectorLite as QAOALiteSelector

def run_benchmark():
    print("="*60)
    print("BENCHMARK INTENSIVO: CLÁSSICO vs QUANTUM V1 vs QUANTUM LITE (V2)")
    print("="*60)
    
    # 1. Carregar Full Dataset
    df = pd.read_csv('data/processed/amazonian_molecules.csv')
    print(f"Dataset Carregado: {len(df)} moléculas.")
    
    # --- PARTE 1: BASELINE CLÁSSICO ("FORÇA TOTAL") ---
    print("\n" + "-"*60)
    print("PARTE 1: REFERÊNCIA CLÁSSICA (GREEDY) - N=1078, K=20")
    print("-" * 60)
    greedy_full = GreedySelector(df)
    s1 = time.time()
    _, div_greedy_full, _ = greedy_full.greedy_selection(k=20)
    t_greedy_full = time.time() - s1
    print(f"  ✓ Greedy Full: {div_greedy_full:.4f} em {t_greedy_full:.4f}s")

    # --- PARTE 2: BATALHA HÍBRIDA QUANTUM (SUBSET LIMIT) ---
    N_subset = 15 # Testando N=15 com a versão Lite
    K_subset = 5
    
    print("\n" + "-"*60)
    print(f"PARTE 2: BATALHA QUÂNTICA - N={N_subset}, K={K_subset}")
    print("-" * 60)
    
    # Sampling justo (seed fixa)
    df_subset = df.sample(n=N_subset, random_state=42).reset_index(drop=True)
    
    # 2.1 QAOA V1 (Nossa Versão)
    print("\n[A] Executando QAOA V1 (Otimizado + Penalty)...")
    try:
        qaoa_v1 = QAOASelector(df_subset, use_real_quantum=False)
        s2 = time.time()
        _, div_qaoa_v1, _ = qaoa_v1.qaoa_optimize(k=K_subset, p=1, shots=1024)
        t_qaoa_v1 = time.time() - s2
        print(f"  ✓ QAOA V1: {div_qaoa_v1:.4f} em {t_qaoa_v1:.4f}s")
    except Exception as e:
        print(f"  ✗ Erro QAOA V1: {e}")
        div_qaoa_v1 = 0
        t_qaoa_v1 = 0

    # 2.2 QAOA LITE (Sua Nova Versão)
    print("\n[B] Executando QAOA LITE (Versão Turbo)...")
    try:
        qaoa_lite = QAOALiteSelector(df_subset)
        s3 = time.time()
        indices_lite, t_internal_lite = qaoa_lite.qaoa_optimize_lite(k=K_subset, p=1)
        t_qaoa_lite = time.time() - s3
        
        # Calcular diversidade
        evaluator = GreedySelector(df_subset)
        div_qaoa_lite = evaluator.calculate_diversity_score(indices_lite)
        print(f"  ✓ QAOA LITE: {div_qaoa_lite:.4f} em {t_qaoa_lite:.4f}s")
    except Exception as e:
        print(f"  ✗ Erro QAOA LITE: {e}")
        import traceback
        traceback.print_exc()
        div_qaoa_lite = 0
        t_qaoa_lite = 0

    # 2.3 Greedy no Subset
    print("\n[C] Executando Greedy no Subset...")
    greedy_subset = GreedySelector(df_subset)
    _, div_greedy_subset, _ = greedy_subset.greedy_selection(k=K_subset)
    print(f"  ✓ Greedy Local: {div_greedy_subset:.4f}")

    # --- RESUMO FINAL ---
    print("\n" + "="*60)
    print("RELATÓRIO FINAL")
    print("="*60)
    print(f"Greedy (Global 1078): {div_greedy_full:.4f}")
    print("-" * 30)
    print(f"Subset (N=15) Benchmark:")
    print(f"{'Algoritmo':<15} | {'Diversidade':<10} | {'Tempo (s)':<10} | {'Eficiência':<10}")
    print("-" * 60)
    print(f"{'Greedy':<15} | {div_greedy_subset:<10.4f} | {'-':<10} | {'100%':<10}")
    
    eff_v1 = (div_qaoa_v1 / div_greedy_subset) * 100 if div_greedy_subset > 0 else 0
    print(f"{'QAOA V1':<15} | {div_qaoa_v1:<10.4f} | {t_qaoa_v1:<10.2f} | {f'{eff_v1:.1f}%':<10}")
    
    eff_lite = (div_qaoa_lite / div_greedy_subset) * 100 if div_greedy_subset > 0 else 0
    print(f"{'QAOA LITE':<15} | {div_qaoa_lite:<10.4f} | {t_qaoa_lite:<10.2f} | {f'{eff_lite:.1f}%':<10}")
    print("="*60)

    # Save Results
    pd.DataFrame([{
        'Algorithm': ['Greedy_Full', 'Greedy_Subset', 'QAOA_V1', 'QAOA_LITE'],
        'Diversity': [div_greedy_full, div_greedy_subset, div_qaoa_v1, div_qaoa_lite],
        'Time': [t_greedy_full, 0, t_qaoa_v1, t_qaoa_lite]
    }]).to_csv('data/results/full_scale_benchmark.csv', index=False)

if __name__ == "__main__":
    run_benchmark()
