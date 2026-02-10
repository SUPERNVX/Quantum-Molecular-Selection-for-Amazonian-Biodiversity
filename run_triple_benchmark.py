import pandas as pd
import numpy as np
import time
import os

# Import dos seletores
from classical_molecular_selection import MolecularDiversitySelector as GreedySelector
from quantum_molecular_selection import QuantumMolecularSelector as QAOASelector
from quantum_molecular_selection_optimized import QuantumMolecularSelector as QAOAOptimizedSelector

def run_benchmark():
    print("="*60)
    print("BENCHMARK TRIPLO: CLÁSSICO vs QUANTUM vs QUANTUM OTIMIZADO")
    print("="*60)
    
    # 1. Carregar dados e preparar subset
    df = pd.read_csv('data/processed/amazonian_molecules.csv')
    # Usar a mesma semente 42 para justiça
    df_subset = df.sample(n=15, random_state=42).reset_index(drop=True)
    K = 5
    
    print(f"Nucleo de teste: N=15, K={K}")
    print("-" * 30)

    # 2. Algoritmo Clássico (Greedy)
    print("[1/3] Executando Clássico (Greedy)...")
    greedy = GreedySelector(df_subset)
    start = time.time()
    _, div_greedy, _ = greedy.greedy_selection(k=K)
    t_greedy = time.time() - start
    print(f"  ✓ Greedy: {div_greedy:.4f} em {t_greedy:.4f}s")

    # 3. QAOA (Nosso Atual)
    print("\n[2/3] Executando QAOA (Nossa Versão)...")
    qaoa = QAOASelector(df_subset, use_real_quantum=False)
    start = time.time()
    _, div_qaoa, _ = qaoa.qaoa_optimize(k=K, p=1, shots=1024)
    t_qaoa = time.time() - start
    print(f"  ✓ QAOA Atual: {div_qaoa:.4f} em {t_qaoa:.4f}s")

    # 4. QAOA Otimizado (Versão do Usuário)
    # Nota: A versão do usuário usa Qiskit V2 Primitives e tem uma lógica de extração diferente
    print("\n[3/3] Executando QAOA Otimizado (Sua Versão)...")
    try:
        qaoa_opt = QAOAOptimizedSelector(df_subset, use_real_quantum=False)
        start = time.time()
        # A API dele retorna (indices, tempo_interno)
        indices_opt, t_opt_internal = qaoa_opt.qaoa_optimize(k=K, p=1, shots=2048)
        t_qaoa_opt = time.time() - start
        
        # Calcular diversidade para os índices retornados usando o sel. clássico
        div_qaoa_opt = greedy._calculate_diversity(indices_opt)
        print(f"  ✓ QAOA Otimizado: {div_qaoa_opt:.4f} em {t_qaoa_opt:.4f}s")
    except Exception as e:
        print(f"  ✗ Erro no QAOA Otimizado: {e}")
        div_qaoa_opt = 0
        t_qaoa_opt = 0

    # 5. Resumo Final
    print("\n" + "="*60)
    print("RESUMO COMPARATIVO")
    print("="*60)
    print(f"{'Algoritmo':<20} | {'Diversidade':<12} | {'Tempo (s)':<10}")
    print("-" * 50)
    print(f"{'Greedy':<20} | {div_greedy:<12.4f} | {t_greedy:<10.4f}")
    print(f"{'QAOA Atual':<20} | {div_qaoa:<12.4f} | {t_qaoa:<10.4f}")
    print(f"{'QAOA Otimizado':<20} | {div_qaoa_opt:<12.4f} | {t_qaoa_opt:<10.4f}")
    print("="*60)

    # Salvar resultados
    res_df = pd.DataFrame({
        'Algoritmo': ['Greedy', 'QAOA Atual', 'QAOA Otimizado'],
        'Diversidade': [div_greedy, div_qaoa, div_qaoa_opt],
        'Tempo': [t_greedy, t_qaoa, t_qaoa_opt]
    })
    res_df.to_csv('data/results/triple_comparison.csv', index=False)

if __name__ == "__main__":
    run_benchmark()
